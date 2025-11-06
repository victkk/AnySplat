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
from src.misc.image_io import save_image

import cv2
from PIL import Image, ImageDraw, ImageFont


def create_comparison_image(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    psnr: float,
    ssim: float,
    lpips: float,
    idx: int
) -> torch.Tensor:
    """
    创建并排对比图，带指标标注

    Args:
        rendered: 渲染图像 [3, H, W]
        gt: ground truth 图像 [3, H, W]
        psnr, ssim, lpips: 指标值
        idx: 图像索引

    Returns:
        comparison: 对比图 [3, H, W*2]
    """
    # 转换为 numpy 数组
    rendered_np = (rendered.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    gt_np = (gt.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # 获取两张图像的尺寸（可能不同）
    h_rendered, w_rendered = rendered_np.shape[:2]
    h_gt, w_gt = gt_np.shape[:2]

    # 使用最大尺寸作为每个图像区域的大小
    h = max(h_rendered, h_gt)
    w = max(w_rendered, w_gt)

    # 创建并排图像，确保宽度足够放下两张图
    # 左边区域 5px + w + 5px，右边区域 5px + w + 5px
    comparison = np.zeros((h + 60, w * 2 + 20, 3), dtype=np.uint8)

    # 放置左边的 rendered 图像（居中对齐）
    y_offset_rendered = (h - h_rendered) // 2
    x_offset_rendered = (w - w_rendered) // 2
    comparison[30+y_offset_rendered:30+y_offset_rendered+h_rendered,
               5+x_offset_rendered:5+x_offset_rendered+w_rendered] = rendered_np

    # 放置右边的 gt 图像（居中对齐）
    y_offset_gt = (h - h_gt) // 2
    x_offset_gt = (w - w_gt) // 2
    comparison[30+y_offset_gt:30+y_offset_gt+h_gt,
               w+15+x_offset_gt:w+15+x_offset_gt+w_gt] = gt_np

    # 转换为 PIL Image 以添加文字
    comparison_pil = Image.fromarray(comparison)
    draw = ImageDraw.Draw(comparison_pil)

    # 尝试使用系统字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font

    # 添加标题
    draw.text((5, 5), f"Rendered (Frame {idx})", fill=(255, 255, 255), font=font)
    draw.text((w+15, 5), "Ground Truth", fill=(255, 255, 255), font=font)

    # 添加指标
    metrics_text = f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | LPIPS: {lpips:.4f}"
    draw.text((5, h + 35), metrics_text, fill=(255, 255, 0), font=font_small)

    # 转换回 tensor
    comparison_tensor = torch.from_numpy(np.array(comparison_pil)).permute(2, 0, 1).float() / 255.0

    return comparison_tensor


def create_error_map(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    创建误差热图

    Args:
        rendered: 渲染图像 [3, H, W]
        gt: ground truth 图像 [3, H, W]

    Returns:
        error_map: 误差热图 [3, H, W]
    """
    # 如果尺寸不匹配，将 rendered 调整到与 gt 相同的尺寸
    if rendered.shape != gt.shape:
        import torch.nn.functional as F
        rendered = F.interpolate(
            rendered.unsqueeze(0),
            size=(gt.shape[1], gt.shape[2]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    # 计算 L1 误差
    error = torch.abs(rendered - gt).mean(dim=0)  # [H, W]

    # 归一化到 [0, 1]
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)

    # 应用 colormap (使用 matplotlib)
    import matplotlib.pyplot as plt
    error_np = error_norm.cpu().numpy()
    error_colored = plt.cm.hot(error_np)[:, :, :3]  # [H, W, 3]

    # 转换回 tensor
    error_tensor = torch.from_numpy(error_colored).permute(2, 0, 1).float()

    return error_tensor


def save_scene_images(
    scene_output_dir: Path,
    rendered_images: torch.Tensor,
    gt_images: torch.Tensor,
    psnr: torch.Tensor,
    ssim: torch.Tensor,
    lpips: torch.Tensor,
    save_comparison: bool = True,
    save_error_map: bool = True
):
    """
    保存场景的所有图像

    Args:
        scene_output_dir: 输出目录
        rendered_images: 渲染图像 [N, 3, H, W]
        gt_images: GT图像 [N, 3, H, W]
        psnr, ssim, lpips: 每张图像的指标
        save_comparison: 是否保存对比图
        save_error_map: 是否保存误差图
    """
    # 创建子目录
    rendered_dir = scene_output_dir / "rendered"
    gt_dir = scene_output_dir / "gt"
    rendered_dir.mkdir(exist_ok=True, parents=True)
    gt_dir.mkdir(exist_ok=True, parents=True)

    if save_comparison:
        comparison_dir = scene_output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True, parents=True)

    if save_error_map:
        error_dir = scene_output_dir / "error_map"
        error_dir.mkdir(exist_ok=True, parents=True)

    # 保存每张图像
    for idx, (rendered, gt, p, s, l) in enumerate(zip(
        rendered_images, gt_images, psnr, ssim, lpips
    )):
        # 基础图像
        save_image(rendered, rendered_dir / f"{idx:04d}.jpg")
        save_image(gt, gt_dir / f"{idx:04d}.jpg")

        # 对比图
        if save_comparison:
            comp = create_comparison_image(rendered, gt, p.item(), s.item(), l.item(), idx)
            save_image(comp, comparison_dir / f"{idx:04d}.jpg")

        # 误差图
        if save_error_map:
            error_map = create_error_map(rendered, gt)
            save_image(error_map, error_dir / f"{idx:04d}.jpg")

    print(f"  保存图像到: {scene_output_dir}")
    print(f"    - 渲染图: {len(rendered_images)} 张")
    if save_comparison:
        print(f"    - 对比图: {len(rendered_images)} 张")
    if save_error_map:
        print(f"    - 误差图: {len(rendered_images)} 张")


def process_image_with_intrinsics(
    image_path: Path,
    intrinsics: np.ndarray,
    original_size: tuple[int, int],
    target_size: tuple[int, int] = (252, 448)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    处理图像并调整相机内参（用于 252×448 分辨率评测）

    从原始 DL3DV images_8 (270×480) center crop 到 252×448，
    并正确调整归一化的相机内参矩阵。

    Args:
        image_path: 图像路径
        intrinsics: 归一化的 3x3 内参矩阵 (已除以原始图像尺寸)
        original_size: 原始图像尺寸 (height, width)
        target_size: 目标尺寸 (height, width)，默认 252×448

    Returns:
        processed_image: 处理后的图像 tensor [3, H, W]，范围 [-1, 1]
        adjusted_intrinsics: 调整后的归一化内参矩阵 [3, 3]
    """
    import torchvision.transforms as transforms

    # 加载图像
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    h_orig, w_orig = original_size
    h_target, w_target = target_size

    # 验证尺寸能被 14 整除（patch size 要求）
    assert h_target % 14 == 0, f"Height {h_target} must be divisible by 14"
    assert w_target % 14 == 0, f"Width {w_target} must be divisible by 14"

    # 获取实际图像尺寸
    img_w, img_h = img.size

    # Center crop 计算
    crop_top = (img_h - h_target) // 2
    crop_left = (img_w - w_target) // 2
    crop_bottom = crop_top + h_target
    crop_right = crop_left + w_target

    # 执行 crop
    img_cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # 转换为 tensor，归一化到 [-1, 1]
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_cropped) * 2.0 - 1.0

    # 调整内参
    # DL3DV 的内参是归一化的：K[i,j] = pixel_value / image_dimension
    intrinsics_adjusted = intrinsics.copy()

    # 1. 反归一化到像素空间（基于原始尺寸）
    fx_pixel = intrinsics[0, 0] * w_orig
    fy_pixel = intrinsics[1, 1] * h_orig
    cx_pixel = intrinsics[0, 2] * w_orig
    cy_pixel = intrinsics[1, 2] * h_orig

    # 2. 调整主点坐标（center crop 的偏移）
    cx_new = cx_pixel - crop_left
    cy_new = cy_pixel - crop_top

    # 3. 重新归一化（基于目标尺寸）
    intrinsics_adjusted[0, 0] = fx_pixel / w_target
    intrinsics_adjusted[1, 1] = fy_pixel / h_target
    intrinsics_adjusted[0, 2] = cx_new / w_target
    intrinsics_adjusted[1, 2] = cy_new / h_target

    return img_tensor, intrinsics_adjusted


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
        '--save_comparison',
        action='store_true',
        default=True,
        help='是否保存并排对比图（rendered vs GT with metrics）'
    )
    parser.add_argument(
        '--save_error_map',
        action='store_true',
        help='是否保存误差热图'
    )
    parser.add_argument(
        '--test_single_scene',
        type=str,
        default=None,
        help='仅测试单个场景（用于调试），提供场景哈希'
    )
    parser.add_argument(
        '--num_scenes',
        type=int,
        default=None,
        help='仅测试前 N 个场景（用于快速验证）。与 --test_single_scene 互斥'
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
    device: torch.device,
    target_size: tuple[int, int] = (252, 448)
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    从数据集加载指定场景和 view indices

    支持 252×448 分辨率评测，正确处理图像 resize 和内参调整

    Args:
        data_root: 数据集根目录
        scene_name: 场景名称（文件夹名）
        context_indices: context view 索引列表
        target_indices: target view 索引列表
        device: 设备
        target_size: 目标图像尺寸 (height, width)，默认 252×448

    Returns:
        context_images, target_images, metadata
    """
    scene_path = data_root / "test" / scene_name

    # 加载 transforms.json
    transforms_path = scene_path / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"未找到 transforms.json: {transforms_path}")

    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)

    # 确定图像文件夹（优先使用 images_8，其次 images_4）
    image_dirs = ["images_8", "images_4", "images"]
    image_dir = None
    for dir_name in image_dirs:
        test_dir = scene_path / dir_name
        if test_dir.exists():
            image_dir = test_dir
            break

    if image_dir is None:
        raise FileNotFoundError(f"未找到图像文件夹: {scene_path}")

    print(f"  使用图像文件夹: {image_dir.name}")

    # 获取原始图像分辨率
    first_image = next(image_dir.iterdir())
    with Image.open(first_image) as img:
        original_w, original_h = img.size
    original_size = (original_h, original_w)

    print(f"  原始分辨率: {original_h}×{original_w}")
    print(f"  目标分辨率: {target_size[0]}×{target_size[1]}")

    # 从 transforms.json 获取内参 (DL3DV 格式)
    h_meta = transforms_data.get('h', original_h)
    w_meta = transforms_data.get('w', original_w)
    fx = transforms_data['fl_x']
    fy = transforms_data['fl_y']
    cx = transforms_data['cx']
    cy = transforms_data['cy']

    # 构建归一化内参矩阵（与 DL3DV dataset 一致）
    base_intrinsics = np.eye(3, dtype=np.float32)
    base_intrinsics[0, 0] = fx / w_meta
    base_intrinsics[1, 1] = fy / h_meta
    base_intrinsics[0, 2] = cx / w_meta
    base_intrinsics[1, 2] = cy / h_meta

    # 加载 context images 和调整内参
    context_images = []
    context_intrinsics = []

    for idx in context_indices:
        frame_info = transforms_data['frames'][idx]
        image_filename = Path(frame_info['file_path']).name
        image_path = image_dir / image_filename

        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")

        # 使用新函数处理图像和内参
        img_tensor, intrinsics_adjusted = process_image_with_intrinsics(
            image_path, base_intrinsics, original_size, target_size
        )

        context_images.append(img_tensor)
        context_intrinsics.append(intrinsics_adjusted)

    # 加载 target images 和调整内参
    target_images = []
    target_intrinsics = []

    for idx in target_indices:
        frame_info = transforms_data['frames'][idx]
        image_filename = Path(frame_info['file_path']).name
        image_path = image_dir / image_filename

        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")

        img_tensor, intrinsics_adjusted = process_image_with_intrinsics(
            image_path, base_intrinsics, original_size, target_size
        )

        target_images.append(img_tensor)
        target_intrinsics.append(intrinsics_adjusted)

    # 转换为 tensor
    context_images = torch.stack(context_images, dim=0).unsqueeze(0).to(device)
    target_images = torch.stack(target_images, dim=0).unsqueeze(0).to(device)

    # 归一化到 [0, 1] (AnySplat 需要)
    context_images = (context_images + 1) * 0.5
    target_images = (target_images + 1) * 0.5

    # 转换内参为 tensor
    context_intrinsics = torch.tensor(
        np.stack(context_intrinsics, axis=0), dtype=torch.float32, device=device
    ).unsqueeze(0)
    target_intrinsics = torch.tensor(
        np.stack(target_intrinsics, axis=0), dtype=torch.float32, device=device
    ).unsqueeze(0)

    metadata = {
        'scene_name': scene_name,
        'image_shape': target_size,
        'original_size': original_size,
        'num_context': len(context_indices),
        'num_target': len(target_indices),
        'context_intrinsics': context_intrinsics,
        'target_intrinsics': target_intrinsics,
    }

    return context_images, target_images, metadata


def evaluate_scene(
    model: AnySplat,
    context_images: torch.Tensor,
    target_images: torch.Tensor,
    metadata: Dict,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    对单个场景进行评测

    支持使用从数据集加载的调整后内参进行渲染

    Args:
        model: AnySplat 模型
        context_images: context 图像 [1, C, 3, H, W]
        target_images: target 图像 [1, T, 3, H, W]
        metadata: 包含调整后的内参和图像尺寸信息
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

    # 4. 获取 Gaussian 基元数量
    # gaussians 通常是一个字典或对象，包含 means, scales, rotations 等
    # 基元数量可以从任何一个属性的第一个维度获取
    num_gaussians = 0
    if hasattr(gaussians, 'means'):
        # 如果是对象属性
        num_gaussians = gaussians.means.shape[1] if gaussians.means.ndim > 1 else gaussians.means.shape[0]
    elif isinstance(gaussians, dict) and 'means' in gaussians:
        # 如果是字典
        num_gaussians = gaussians['means'].shape[1] if gaussians['means'].ndim > 1 else gaussians['means'].shape[0]
    elif hasattr(gaussians, '__len__'):
        # 如果是 tuple 或 list，通常第一个元素是 means
        if len(gaussians) > 0:
            first_elem = gaussians[0]
            num_gaussians = first_elem.shape[1] if first_elem.ndim > 1 else first_elem.shape[0]

    # 5. 计算指标
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
        'num_gaussians': num_gaussians,
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
    # 如果指定测试前 N 个场景
    elif args.num_scenes is not None:
        if args.num_scenes <= 0:
            raise ValueError(f"--num_scenes 必须是正整数，当前值: {args.num_scenes}")

        # 获取前 N 个场景（保持原有顺序）
        scene_keys = list(eval_indices.keys())[:args.num_scenes]
        eval_indices = {k: eval_indices[k] for k in scene_keys}
        print(f"\n仅测试前 {len(eval_indices)} 个场景（共 {args.num_scenes} 个）")

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
    summary_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'num_gaussians': []}

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
            # 加载数据（使用 252×448 分辨率）
            context_images, target_images, metadata = load_scene_from_dataset(
                data_root, scene_name, context_indices, target_indices, device,
                target_size=(252, 448)
            )

            # 评测（传递 metadata）
            results = evaluate_scene(model, context_images, target_images, metadata, device)

            # 保存结果
            scene_metrics = {
                'psnr_mean': results['psnr'].mean().item(),
                'ssim_mean': results['ssim'].mean().item(),
                'lpips_mean': results['lpips'].mean().item(),
                'num_gaussians': results['num_gaussians'],
                'psnr_per_image': results['psnr'].tolist(),
                'ssim_per_image': results['ssim'].tolist(),
                'lpips_per_image': results['lpips'].tolist(),
            }
            all_results[scene_hash] = scene_metrics

            # 累积统计
            summary_metrics['psnr'].append(results['psnr'].mean().item())
            summary_metrics['ssim'].append(results['ssim'].mean().item())
            summary_metrics['lpips'].append(results['lpips'].mean().item())
            summary_metrics['num_gaussians'].append(results['num_gaussians'])

            # 保存图像
            if args.save_images:
                scene_output_dir = output_dir / "images" / scene_name
                save_scene_images(
                    scene_output_dir,
                    results['rendered'],
                    results['gt'],
                    results['psnr'],
                    results['ssim'],
                    results['lpips'],
                    save_comparison=args.save_comparison,
                    save_error_map=args.save_error_map
                )

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
        'num_gaussians': {
            'mean': np.mean(summary_metrics['num_gaussians']),
            'std': np.std(summary_metrics['num_gaussians']),
            'min': int(np.min(summary_metrics['num_gaussians'])),
            'max': int(np.max(summary_metrics['num_gaussians'])),
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
    print(f"Gaussians: {summary['num_gaussians']['mean']:.0f} ± {summary['num_gaussians']['std']:.0f} "
          f"(min: {summary['num_gaussians']['min']}, max: {summary['num_gaussians']['max']})")
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
