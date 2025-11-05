# DL3DV 数据集评测使用指南

本指南说明如何使用统一的评测脚本在 DL3DV 数据集上对比 AnySplat 和 SparseSplat 的性能。

## 快速开始

### 前提条件

1. 已安装 AnySplat 和 SparseSplat 的依赖
2. 有可访问的 DL3DV 数据集
3. 有 SparseSplat 的 evaluation JSON 文件

### 基本用法

#### 在 AnySplat 上评测

```bash
cd AnySplat

# 完整评测（所有 140 个测试场景）
python src/eval_dl3dv_unified.py \
  --data_root /path/to/dl3dv \
  --eval_json ../SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  --output_dir outputs/anysplat_dl3dv_6views \
  --save_images

# 单场景测试（用于调试）
python src/eval_dl3dv_unified.py \
  --data_root /path/to/dl3dv \
  --eval_json ../SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  --output_dir outputs/debug \
  --test_single_scene 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7
```

#### 在 SparseSplat 上评测（参考）

```bash
cd SparseSplat

python -m src.main +experiment=dl3dv \
  dataset.roots=[/path/to/dl3dv] \
  dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  dataset.view_sampler.num_context_views=6 \
  mode=test \
  output_dir=outputs/sparsesplat_dl3dv_6views
```

## 场景映射问题

### 问题说明

SparseSplat 的 evaluation JSON 使用 SHA256 哈希作为场景标识符：
```json
{
  "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7": {
    "context": [0, 19, 29, 33, 39, 49],
    "target": [0, 1, 2, ..., 49]
  }
}
```

而数据集中的场景使用文件夹名。评测脚本会尝试自动建立映射关系。

### 自动映射

脚本会尝试以下方法：
1. **方法1**: 计算场景名的 SHA256，匹配 evaluation JSON 中的哈希
2. **方法2**: 如果数量相等，按字母顺序一一对应（最后的备选方案）

首次运行后，会在输出目录生成 `scene_mapping.json`，后续可重复使用：

```bash
python src/eval_dl3dv_unified.py \
  --data_root /path/to/dl3dv \
  --eval_json path/to/eval.json \
  --scene_mapping outputs/anysplat_dl3dv_6views/scene_mapping.json \
  --output_dir outputs/anysplat_dl3dv_6views
```

### 手动创建映射（如果自动映射失败）

如果自动映射失败，需要手动创建映射文件：

```json
{
  "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7": "scene_folder_name_1",
  "0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166": "scene_folder_name_2",
  ...
}
```

**创建方法**:

1. 从 SparseSplat 的 .torch chunks 中提取场景名：
```bash
cd SparseSplat
python -c "
import torch
import json
from pathlib import Path

# 加载一个 test chunk
chunk_path = Path('datasets/dl3dv_processed/test/000000.torch')
chunk = torch.load(chunk_path)

# 提取所有场景 key
scene_keys = [example['key'] for example in chunk]
print(json.dumps(scene_keys, indent=2))
"
```

2. 对每个场景名计算 SHA256：
```python
import hashlib
import json

scene_names = ["scene1", "scene2", ...]  # 从上一步获取
mapping = {}

for scene_name in scene_names:
    scene_hash = hashlib.sha256(scene_name.encode()).hexdigest()
    mapping[scene_hash] = scene_name

with open('scene_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2)
```

## 输出说明

### 目录结构

```
outputs/anysplat_dl3dv_6views/
├── results.json              # 详细结果
├── scene_mapping.json        # 场景映射（首次运行生成）
└── images/                   # 渲染图像（如果使用 --save_images）
    ├── scene1/
    │   ├── rendered_0000.jpg
    │   ├── gt_0000.jpg
    │   ├── rendered_0001.jpg
    │   └── ...
    └── ...
```

### results.json 格式

```json
{
  "summary": {
    "psnr": {
      "mean": 24.5,
      "std": 2.1,
      "min": 20.3,
      "max": 28.9
    },
    "ssim": {
      "mean": 0.83,
      "std": 0.05,
      "min": 0.75,
      "max": 0.90
    },
    "lpips": {
      "mean": 0.15,
      "std": 0.03,
      "min": 0.08,
      "max": 0.25
    },
    "num_scenes": 140
  },
  "per_scene": {
    "scene_hash_1": {
      "psnr_mean": 25.3,
      "ssim_mean": 0.85,
      "lpips_mean": 0.12,
      "psnr_per_image": [24.5, 25.1, ...],
      "ssim_per_image": [0.84, 0.86, ...],
      "lpips_per_image": [0.13, 0.11, ...]
    }
  }
}
```

## 常见问题

### Q1: 如何验证使用了正确的 view indices?

**A**: 脚本会输出详细日志，包括加载的 context 和 target indices。检查输出：
```
正在评测场景: scene_name
  Context indices: [0, 19, 29, 33, 39, 49]
  Target indices: [0, 1, 2, ..., 49]
```

### Q2: 图像分辨率不匹配怎么办?

**A**: 脚本会自动检测并使用可用的图像文件夹（优先级：images_4 > images_8 > images）。
- `images_4`: 540x960
- `images_8`: 270x480

确保数据集包含这些文件夹之一。

### Q3: 如何确保与 SparseSplat 的结果可比?

**A**: 关键检查点：
1. ✅ 使用相同的 evaluation JSON
2. ✅ 使用相同的图像分辨率
3. ✅ 验证 context/target indices 相同
4. ✅ 检查相机参数格式（归一化方式）

### Q4: 评测太慢怎么办?

**A**:
1. 使用 `--test_single_scene` 先测试单个场景
2. 不使用 `--save_images` 跳过图像保存
3. 使用更少的 target views（修改 evaluation JSON）

### Q5: 遇到 CUDA OOM 错误?

**A**:
- 减少 batch size（代码中固定为 1，但可修改）
- 降低图像分辨率（使用 images_8 而非 images_4）
- 分批处理场景

## 进阶用法

### 比较不同配置

```bash
# 6 context views
python src/eval_dl3dv_unified.py \
  --eval_json path/to/ctx_6v.json \
  --output_dir outputs/6views

# 12 context views
python src/eval_dl3dv_unified.py \
  --eval_json path/to/ctx_12v.json \
  --output_dir outputs/12views
```

### 批量分析结果

```python
import json
import pandas as pd

# 加载多个结果
results_6v = json.load(open('outputs/6views/results.json'))
results_12v = json.load(open('outputs/12views/results.json'))

# 创建对比表格
comparison = pd.DataFrame({
    '6 views': results_6v['summary'],
    '12 views': results_12v['summary']
})
print(comparison)
```

## 数据集准备

### DL3DV 数据结构

确保数据集结构如下：
```
dl3dv/
├── train/
│   ├── scene1/
│   │   ├── transforms.json
│   │   ├── images_4/
│   │   └── images_8/
│   └── ...
└── test/
    ├── scene1/
    │   ├── transforms.json
    │   ├── images_4/
    │   └── images_8/
    └── ...
```

### 如果只有原始数据

如果只有原始 DL3DV 数据，需要：

1. 按照 DL3DV 官方说明下载数据
2. 生成降采样图像：
```bash
python scripts/downsample_images.py \
  --input_dir path/to/original \
  --output_dir path/to/processed \
  --scales 4 8
```

## 与 SparseSplat 结果对比

评测完成后，收集两个项目的结果进行对比：

```python
import json
import matplotlib.pyplot as plt

# 加载结果
anysplat = json.load(open('outputs/anysplat_dl3dv_6views/results.json'))
sparsesplat = json.load(open('outputs/sparsesplat_dl3dv_6views/results.json'))

# 对比汇总指标
metrics = ['psnr', 'ssim', 'lpips']
anysplat_means = [anysplat['summary'][m]['mean'] for m in metrics]
sparsesplat_means = [sparsesplat['summary'][m]['mean'] for m in metrics]

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(metrics):
    axes[i].bar(['AnySplat', 'SparseSplat'],
                [anysplat_means[i], sparsesplat_means[i]])
    axes[i].set_title(metric.upper())
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.savefig('comparison.png')
print("对比图已保存到 comparison.png")
```

## 时间估算

- **单场景**: ~5-10 秒
- **140 场景**: ~15-30 分钟（单 GPU）
- 使用 `--save_images` 会增加 ~20% 时间

## 故障排查

### 问题：找不到 transforms.json

**原因**: 场景文件夹结构不正确

**解决**: 检查场景路径，确保包含 transforms.json

### 问题：图像加载失败

**原因**: 图像路径在 transforms.json 中不正确

**解决**: 脚本会自动尝试从 images_* 文件夹加载，如果仍失败，检查文件名格式

### 问题：场景映射失败

**原因**: 无法自动确定哈希和场景名的对应关系

**解决**:
1. 检查数据集是否完整（140 个测试场景）
2. 手动创建 scene_mapping.json（见上文）
3. 从 SparseSplat 的 .torch chunks 中提取场景信息

## 支持

如有问题，请：
1. 检查本指南的常见问题部分
2. 查看详细的错误日志
3. 使用 `--test_single_scene` 隔离问题
4. 确认数据集和依赖完整性

## 参考

- SparseSplat: `/data/zhangzicheng/workspace/SparseSplat-/SparseSplat`
- AnySplat: `/data/zhangzicheng/workspace/SparseSplat-/AnySplat`
- 方案文档: `/data/zhangzicheng/workspace/SparseSplat-/DL3DV_EVALUATION_PLAN.md`
