"""
DL3DV 统一评测脚本 - 用于与 SparseSplat 进行一致的对比评测

使用 SparseSplat 的 evaluation JSON 来确保使用相同的 view pairs
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.model.model.anysplat import AnySplat
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.image import process_image
from misc.image_io import save_image


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description='在 DL3DV 数据集上评测 AnySplat，使用与 SparseSplat 相同的 evaluation indices'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='DL3DV 数据集根目录（包含 train/ 和 test/ 文件夹）'
    )
    parser.add_argument(
        '--eval_json',
        type=str,
        required=True,
        help='SparseSplat 的 evaluation JSON 文件路径（例如 dl3dv_start_0_distance_50_ctx_6v_video_0_50.json）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/dl3dv_eval',
        help='输出目录'
    )
    parser.add_argument(
        '--scene_mapping',
        type=str,
        default=None,
        help='场景哈希到路径的映射文件（JSON格式）。如果不提供，将尝试自动生成'
    )
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='是否保存渲染图像'
    )
    parser.add_argument(
        '--test_single_scene',
        type=str,
        default=None,
        help='仅测试单个场景（用于调试），提供场景哈希'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='运行设备'
    )

    return parser.parse_args()


def build_scene_mapping_from_dataset(data_root: Path, eval_json_path: Path) -> Dict[str, str]:
    """
    从 DL3DV 数据集构建场景哈希映射

    策略1: 尝试读取 test_index.json（如果存在）
    策略2: 遍历 test 文件夹，按字母顺序对应

    Args:
        data_root: DL3DV 数据集根目录
        eval_json_path: evaluation JSON 路径

    Returns:
        Dict[scene_hash, scene_path]: 场景哈希到路径的映射
    """
    print("正在构建场景映射...")

    # 加载 evaluation JSON 获取所有场景哈希
    with open(eval_json_path, 'r') as f:
        eval_data = json.load(f)
    scene_hashes = list(eval_data.keys())
    print(f"Evaluation JSON 包含 {len(scene_hashes)} 个场景")

    test_dir = data_root / "test"

    # 策略1: 尝试读取 test_index.json
    test_index_path = test_dir / "test_index.json"
    if test_index_path.exists():
        print(f"找到 test_index.json: {test_index_path}")
        with open(test_index_path, 'r') as f:
            scene_list = json.load(f)
        print(f"test_index.json 包含 {len(scene_list)} 个场景")
    else:
        # 策略2: 遍历 test 文件夹
        print(f"未找到 test_index.json，遍历文件夹: {test_dir}")
        scene_list = sorted([
            d.name for d in test_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        print(f"找到 {len(scene_list)} 个场景文件夹")

    # 检查数量是否匹配
    if len(scene_list) != len(scene_hashes):
        print(f"WARNING: 场景数量不匹配！")
        print(f"  Evaluation JSON: {len(scene_hashes)} 个场景")
        print(f"  数据集: {len(scene_list)} 个场景")

    # 尝试多种哈希生成方式
    mapping = {}

    # 方法1: 直接使用场景名的 SHA256
    print("\n尝试方法1: SHA256(scene_name)")
    for scene_path in scene_list:
        scene_hash = hashlib.sha256(scene_path.encode()).hexdigest()
        if scene_hash in scene_hashes:
            mapping[scene_hash] = scene_path
    print(f"  匹配: {len(mapping)}/{len(scene_hashes)}")

    if len(mapping) == len(scene_hashes):
        print("✓ 成功建立完整映射！")
        return mapping

    # 方法2: 如果数量相等，按顺序对应（最后的备选方案）
    if len(scene_list) == len(scene_hashes):
        print(f"\n尝试方法2: 按字母顺序一一对应")
        mapping = {
            hash_key: scene_path
            for hash_key, scene_path in zip(sorted(scene_hashes), sorted(scene_list))
        }
        print(f"  强制映射: {len(mapping)} 个场景")
        print("  WARNING: 这是基于顺序的假设，可能不准确！")
        return mapping

    raise ValueError(
        f"无法建立场景映射！请手动提供 --scene_mapping 文件。\n"
        f"映射文件格式: {{'scene_hash': 'scene_folder_name', ...}}"
    )


def save_scene_mapping(mapping: Dict[str, str], output_path: Path):
    """保存场景映射到文件"""
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"场景映射已保存到: {output_path}")


def load_scene_from_dataset(
    data_root: Path,
    scene_name: str,
    context_indices: List[int],
    target_indices: List[int],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    从数据集加载指定场景和 view indices

    Args:
        data_root: 数据集根目录
        scene_name: 场景名称（文件夹名）
        context_indices: context view 索引列表
        target_indices: target view 索引列表
        device: 设备

    Returns:
        context_images, target_images, metadata
    """
    from PIL import Image

    scene_path = data_root / "test" / scene_name

    # 加载 transforms.json
    transforms_path = scene_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"未找到 transforms.json: {transforms_path}")

    with open(transforms_path, 'r') as f:
        transforms = json.load(f)

    # 确定图像文件夹（优先使用 images_4，其次 images_8）
    image_dirs = ["images_4", "images_8", "images"]
    image_dir = None
    for dir_name in image_dirs:
        test_dir = scene_path / dir_name
        if test_dir.exists():
            image_dir = test_dir
            break

    if image_dir is None:
        raise FileNotFoundError(f"未找到图像文件夹: {scene_path}")

    print(f"  使用图像文件夹: {image_dir.name}")

    # 获取图像分辨率
    first_image = next(image_dir.iterdir())
    with Image.open(first_image) as img:
        image_w, image_h = img.size
    print(f"  图像分辨率: {image_h}x{image_w}")

    # 加载 context images
    context_images = []
    for idx in context_indices:
        frame_info = transforms['frames'][idx]
        image_path = scene_path / frame_info['file_path']
        # 如果路径不存在，尝试从 image_dir 加载
        if not image_path.exists():
            filename = Path(frame_info['file_path']).name
            image_path = image_dir / filename

        image = process_image(str(image_path))
        context_images.append(image)

    # 加载 target images
    target_images = []
    for idx in target_indices:
        frame_info = transforms['frames'][idx]
        image_path = scene_path / frame_info['file_path']
        if not image_path.exists():
            filename = Path(frame_info['file_path']).name
            image_path = image_dir / filename

        image = process_image(str(image_path))
        target_images.append(image)

    context_images = torch.stack(context_images, dim=0).unsqueeze(0).to(device)
    target_images = torch.stack(target_images, dim=0).unsqueeze(0).to(device)

    # 归一化到 [0, 1]
    context_images = (context_images + 1) * 0.5
    target_images = (target_images + 1) * 0.5

    metadata = {
        'scene_name': scene_name,
        'image_shape': (image_h, image_w),
        'num_context': len(context_indices),
        'num_target': len(target_indices),
    }

    return context_images, target_images, metadata


def evaluate_scene(
    model: AnySplat,
    context_images: torch.Tensor,
    target_images: torch.Tensor,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    对单个场景进行评测

    Args:
        model: AnySplat 模型
        context_images: context 图像 [1, C, 3, H, W]
        target_images: target 图像 [1, T, 3, H, W]
        device: 设备

    Returns:
        dict: {
            'psnr': per-image PSNR,
            'ssim': per-image SSIM,
            'lpips': per-image LPIPS,
            'rendered': 渲染图像
        }
    """
    b, num_context, _, h, w = context_images.shape
    num_target = target_images.shape[1]

    # 1. Encoder forward
    with torch.no_grad():
        encoder_output = model.encoder(
            context_images,
            global_step=0,
            visualization_dump={},
        )
        gaussians = encoder_output.gaussians
        pred_context_pose = encoder_output.pred_context_pose

    # 2. 预测所有 poses (context + target)
    all_images = torch.cat((context_images, target_images), dim=1).to(torch.bfloat16)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(
            all_images,
            intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx
        )

    with torch.cuda.amp.autocast(enabled=False):
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
            pred_all_pose_enc,
            all_images.shape[-2:]
        )

    # 添加 padding 并转换为 c2w
    extrinsic_padding = torch.tensor(
        [0, 0, 0, 1],
        device=pred_all_extrinsic.device,
        dtype=pred_all_extrinsic.dtype
    ).view(1, 1, 1, 4).repeat(b, all_images.shape[1], 1, 1)
    pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

    # 归一化 intrinsics
    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h

    # 分离 context 和 target poses
    pred_context_extrinsic = pred_all_extrinsic[:, :num_context]
    pred_target_extrinsic = pred_all_extrinsic[:, num_context:]
    pred_context_intrinsic = pred_all_intrinsic[:, :num_context]
    pred_target_intrinsic = pred_all_intrinsic[:, num_context:]

    # 计算 scale factor
    scale_factor = (
        pred_context_pose['extrinsic'][:, :, :3, 3].mean() /
        pred_context_extrinsic[:, :, :3, 3].mean()
    )
    pred_target_extrinsic[..., :3, 3] = pred_target_extrinsic[..., :3, 3] * scale_factor

    # 3. Decoder forward
    with torch.no_grad():
        output = model.decoder.forward(
            gaussians,
            pred_target_extrinsic,
            pred_target_intrinsic.float(),
            torch.ones(1, num_target, device=device) * 0.01,
            torch.ones(1, num_target, device=device) * 100,
            (h, w)
        )

    # 4. 计算指标
    rendered_images = output.color[0]
    gt_images = target_images[0]

    psnr = compute_psnr(rendered_images, gt_images)
    ssim = compute_ssim(rendered_images, gt_images)
    lpips = compute_lpips(rendered_images, gt_images)

    return {
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips,
        'rendered': rendered_images,
        'gt': gt_images,
    }


def main():
    args = setup_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 加载 evaluation indices
    print(f"\n加载 evaluation indices: {args.eval_json}")
    with open(args.eval_json, 'r') as f:
        eval_indices = json.load(f)
    print(f"找到 {len(eval_indices)} 个场景")

    # 建立场景映射
    data_root = Path(args.data_root)
    if args.scene_mapping:
        print(f"\n加载场景映射: {args.scene_mapping}")
        with open(args.scene_mapping, 'r') as f:
            scene_mapping = json.load(f)
    else:
        scene_mapping = build_scene_mapping_from_dataset(data_root, Path(args.eval_json))
        # 保存映射供后续使用
        mapping_path = output_dir / "scene_mapping.json"
        save_scene_mapping(scene_mapping, mapping_path)

    # 如果指定单个场景测试
    if args.test_single_scene:
        eval_indices = {args.test_single_scene: eval_indices[args.test_single_scene]}
        print(f"\n仅测试场景: {args.test_single_scene}")

    # 加载模型
    print(f"\n加载 AnySplat 模型...")
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("模型加载完成")

    # 评测所有场景
    all_results = {}
    summary_metrics = {'psnr': [], 'ssim': [], 'lpips': []}

    print(f"\n开始评测 {len(eval_indices)} 个场景...")
    for scene_hash, indices in tqdm(eval_indices.items(), desc="评测进度"):
        if indices is None:
            print(f"跳过场景 {scene_hash}: 无有效 indices")
            continue

        # 获取场景路径
        if scene_hash not in scene_mapping:
            print(f"WARNING: 未找到场景映射: {scene_hash}")
            continue
        scene_name = scene_mapping[scene_hash]

        context_indices = indices['context']
        target_indices = indices['target']

        try:
            # 加载数据
            context_images, target_images, metadata = load_scene_from_dataset(
                data_root, scene_name, context_indices, target_indices, device
            )

            # 评测
            results = evaluate_scene(model, context_images, target_images, device)

            # 保存结果
            scene_metrics = {
                'psnr_mean': results['psnr'].mean().item(),
                'ssim_mean': results['ssim'].mean().item(),
                'lpips_mean': results['lpips'].mean().item(),
                'psnr_per_image': results['psnr'].tolist(),
                'ssim_per_image': results['ssim'].tolist(),
                'lpips_per_image': results['lpips'].tolist(),
            }
            all_results[scene_hash] = scene_metrics

            # 累积统计
            summary_metrics['psnr'].append(results['psnr'].mean().item())
            summary_metrics['ssim'].append(results['ssim'].mean().item())
            summary_metrics['lpips'].append(results['lpips'].mean().item())

            # 保存图像
            if args.save_images:
                scene_output_dir = output_dir / "images" / scene_name
                scene_output_dir.mkdir(exist_ok=True, parents=True)
                for idx, (rendered, gt) in enumerate(zip(results['rendered'], results['gt'])):
                    save_image(rendered, scene_output_dir / f"rendered_{idx:04d}.jpg")
                    save_image(gt, scene_output_dir / f"gt_{idx:04d}.jpg")

        except Exception as e:
            print(f"ERROR 处理场景 {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 计算汇总统计
    summary = {
        'psnr': {
            'mean': np.mean(summary_metrics['psnr']),
            'std': np.std(summary_metrics['psnr']),
            'min': np.min(summary_metrics['psnr']),
            'max': np.max(summary_metrics['psnr']),
        },
        'ssim': {
            'mean': np.mean(summary_metrics['ssim']),
            'std': np.std(summary_metrics['ssim']),
            'min': np.min(summary_metrics['ssim']),
            'max': np.max(summary_metrics['ssim']),
        },
        'lpips': {
            'mean': np.mean(summary_metrics['lpips']),
            'std': np.std(summary_metrics['lpips']),
            'min': np.min(summary_metrics['lpips']),
            'max': np.max(summary_metrics['lpips']),
        },
        'num_scenes': len(summary_metrics['psnr']),
    }

    # 打印结果
    print("\n" + "="*50)
    print("评测结果汇总")
    print("="*50)
    print(f"场景数量: {summary['num_scenes']}")
    print(f"PSNR:  {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f} "
          f"(min: {summary['psnr']['min']:.2f}, max: {summary['psnr']['max']:.2f})")
    print(f"SSIM:  {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f} "
          f"(min: {summary['ssim']['min']:.4f}, max: {summary['ssim']['max']:.4f})")
    print(f"LPIPS: {summary['lpips']['mean']:.4f} ± {summary['lpips']['std']:.4f} "
          f"(min: {summary['lpips']['min']:.4f}, max: {summary['lpips']['max']:.4f})")
    print("="*50)

    # 保存结果
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'summary': summary,
            'per_scene': all_results,
        }, f, indent=2)
    print(f"\n详细结果已保存到: {results_path}")


if __name__ == "__main__":
    main()
