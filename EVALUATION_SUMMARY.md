# DL3DV 评测对比方案总结

## 方案概述

为了对比 AnySplat 和 SparseSplat 在 DL3DV 数据集 6 视角下的渲染指标，我已经创建了一套完整的评测方案。

## 核心思路

1. **使用统一的 evaluation indices** - 复用 SparseSplat 的 evaluation JSON 文件，确保两个方法使用完全相同的 view pairs
2. **最小化代码修改** - 在 AnySplat 中添加新的评测脚本，不影响原有代码
3. **保证可复现性** - 固定的 context/target view indices，确保结果可对比

## 已创建的文件

### 1. 详细方案文档
**文件**: `/data/zhangzicheng/workspace/SparseSplat-/DL3DV_EVALUATION_PLAN.md`

包含：
- 两个项目的核心差异分析
- 详细的实施步骤
- 场景哈希映射策略
- 技术细节和注意事项

### 2. 评测脚本
**文件**: `/data/zhangzicheng/workspace/SparseSplat-/AnySplat/src/eval_dl3dv_unified.py`

功能：
- ✅ 读取 SparseSplat 的 evaluation JSON
- ✅ 自动建立场景哈希映射
- ✅ 使用 AnySplat 模型进行推理
- ✅ 计算 PSNR、SSIM、LPIPS 指标
- ✅ 支持单场景调试
- ✅ 保存详细结果和可选的渲染图像

### 3. 使用指南
**文件**: `/data/zhangzicheng/workspace/SparseSplat-/EVALUATION_USAGE_GUIDE.md`

包含：
- 快速开始教程
- 场景映射问题解决方案
- 常见问题和故障排查
- 结果对比方法

## 快速使用

### 在 AnySplat 上评测

```bash
cd /data/zhangzicheng/workspace/SparseSplat-/AnySplat

# 完整评测
python src/eval_dl3dv_unified.py \
  --data_root /path/to/dl3dv \
  --eval_json ../SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  --output_dir outputs/anysplat_dl3dv_6v

# 单场景测试（调试）
python src/eval_dl3dv_unified.py \
  --data_root /path/to/dl3dv \
  --eval_json ../SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  --test_single_scene 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7 \
  --output_dir outputs/debug
```

### 在 SparseSplat 上评测（参考）

```bash
cd /data/zhangzicheng/workspace/SparseSplat-/SparseSplat

python -m src.main +experiment=dl3dv \
  dataset.roots=[/path/to/dl3dv] \
  dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  mode=test \
  output_dir=outputs/sparsesplat_dl3dv_6v
```

## 关键技术点

### 1. 场景映射问题

**问题**: SparseSplat 的 evaluation JSON 使用 SHA256 哈希作为场景标识符，而数据集使用文件夹名。

**解决**: 评测脚本会尝试：
- 方法1: 计算场景名的 SHA256 进行匹配
- 方法2: 按字母顺序对应（备选方案）
- 首次运行后生成 `scene_mapping.json` 供后续使用

如果自动映射失败，可以手动创建映射文件或从 SparseSplat 的 .torch chunks 提取场景信息。

### 2. 数据加载

脚本会：
- 从 transforms.json 读取相机参数
- 自动检测可用的图像文件夹（images_4 优先，其次 images_8）
- 根据 evaluation JSON 中的 indices 选择对应的 views

### 3. 推理流程

与 AnySplat 的 `eval_nvs.py` 保持一致：
1. Encoder: 从 context images 生成 Gaussians
2. Camera prediction: 使用 VGGT 预测所有 poses
3. Scale alignment: 对齐 predicted 和 encoder 的 poses
4. Decoder: 渲染 target views
5. Metrics: 计算 PSNR、SSIM、LPIPS

## 评测数据

- **数据集**: DL3DV test split
- **场景数量**: 140 个场景
- **Context views**: 6 个固定 views（由 evaluation JSON 指定）
- **Target views**: 每个场景 50 个 views（通常是 0-49）
- **图像分辨率**: 270x480 (images_8) 或 540x960 (images_4)

## 输出结果

### 结果文件 (results.json)

```json
{
  "summary": {
    "psnr": {"mean": 24.5, "std": 2.1, "min": 20.3, "max": 28.9},
    "ssim": {"mean": 0.83, "std": 0.05, "min": 0.75, "max": 0.90},
    "lpips": {"mean": 0.15, "std": 0.03, "min": 0.08, "max": 0.25},
    "num_scenes": 140
  },
  "per_scene": {
    "scene_hash": {
      "psnr_mean": 25.3,
      "ssim_mean": 0.85,
      "lpips_mean": 0.12,
      "psnr_per_image": [...],
      "ssim_per_image": [...],
      "lpips_per_image": [...]
    }
  }
}
```

## 下一步建议

### 1. 立即执行（必需）

1. **验证数据集路径和结构**
   ```bash
   ls -la /path/to/dl3dv/test/  # 确认有 140 个场景文件夹
   ```

2. **单场景测试**
   ```bash
   # 先在单个场景上验证流程
   python src/eval_dl3dv_unified.py \
     --data_root /path/to/dl3dv \
     --eval_json ../SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
     --test_single_scene <first_scene_hash> \
     --output_dir outputs/debug \
     --save_images
   ```

3. **检查场景映射**
   - 查看生成的 `scene_mapping.json` 是否合理
   - 验证映射的场景数量是否为 140
   - 如果自动映射失败，需要手动创建映射（见使用指南）

### 2. 完整评测（验证通过后）

1. **AnySplat 完整评测**
   ```bash
   python src/eval_dl3dv_unified.py \
     --data_root /path/to/dl3dv \
     --eval_json ../SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
     --output_dir outputs/anysplat_dl3dv_6v
   ```

2. **SparseSplat 完整评测**（如果还没运行）
   ```bash
   cd ../SparseSplat
   python -m src.main +experiment=dl3dv \
     dataset.roots=[/path/to/dl3dv] \
     dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
     mode=test \
     output_dir=outputs/sparsesplat_dl3dv_6v
   ```

### 3. 结果分析

1. **汇总对比**
   ```python
   import json

   anysplat = json.load(open('outputs/anysplat_dl3dv_6v/results.json'))
   sparsesplat = json.load(open('outputs/sparsesplat_dl3dv_6v/results.json'))

   print("AnySplat:")
   print(f"  PSNR: {anysplat['summary']['psnr']['mean']:.2f}")
   print(f"  SSIM: {anysplat['summary']['ssim']['mean']:.4f}")
   print(f"  LPIPS: {anysplat['summary']['lpips']['mean']:.4f}")

   print("\nSparseSplat:")
   print(f"  PSNR: {sparsesplat['summary']['psnr']['mean']:.2f}")
   print(f"  SSIM: {sparsesplat['summary']['ssim']['mean']:.4f}")
   print(f"  LPIPS: {sparsesplat['summary']['lpips']['mean']:.4f}")
   ```

2. **可视化对比**（如果需要）
   - 对比渲染图像质量
   - 分析 per-scene 结果差异
   - 找出表现差异大的场景

## 潜在问题和解决方案

### 问题 1: 场景映射失败

**症状**: 脚本报错 "无法建立场景映射"

**解决步骤**:
1. 检查 DL3DV 数据集是否完整（140 个测试场景）
2. 尝试从 SparseSplat 的 .torch chunks 提取场景名：
   ```python
   import torch
   chunk = torch.load('SparseSplat/datasets/dl3dv_processed/test/000000.torch')
   scene_keys = [ex['key'] for ex in chunk]
   print(scene_keys)
   ```
3. 手动创建 scene_mapping.json（参见使用指南）

### 问题 2: 图像分辨率不匹配

**症状**: 加载图像失败或分辨率错误

**解决**: 确保数据集包含 `images_4` 或 `images_8` 文件夹。如果只有原始图像，需要降采样。

### 问题 3: CUDA OOM

**症状**: GPU 内存不足

**解决**:
- 使用更低分辨率的图像（images_8 而非 images_4）
- 分批处理场景
- 减少 target views 数量

### 问题 4: 结果差异很大

**症状**: AnySplat 和 SparseSplat 的结果相差很大

**检查**:
1. 是否使用了相同的 view indices？
2. 图像分辨率是否一致？
3. 相机参数归一化方式是否相同？
4. 深度范围（near/far）是否一致？

## 预期时间开销

- **单场景测试**: ~5-10 秒
- **完整评测（140 场景）**: ~15-30 分钟
- **总调试+评测时间**: 1-2 小时（假设一切顺利）

## 技术支持

如遇到问题：
1. 查看 `EVALUATION_USAGE_GUIDE.md` 的常见问题部分
2. 使用 `--test_single_scene` 隔离问题
3. 检查详细的错误日志和堆栈跟踪
4. 验证数据集和依赖完整性

## 文件清单

```
/data/zhangzicheng/workspace/SparseSplat-/
├── DL3DV_EVALUATION_PLAN.md          # 详细技术方案
├── EVALUATION_USAGE_GUIDE.md         # 使用指南
├── EVALUATION_SUMMARY.md             # 本文件 - 快速总结
├── AnySplat/
│   └── src/
│       └── eval_dl3dv_unified.py     # 评测脚本
└── SparseSplat/
    └── assets/
        └── dl3dv_start_0_distance_50_ctx_6v_video_0_50.json  # Evaluation indices
```

## 总结

✅ **方案完整性**: 提供了从理论分析到实际实现的完整方案
✅ **可复现性**: 使用固定的 evaluation indices 确保结果可对比
✅ **易用性**: 提供了详细的使用指南和故障排查
✅ **灵活性**: 支持单场景调试和完整评测

现在你可以开始运行评测了！建议先进行单场景测试验证流程，然后再运行完整评测。
