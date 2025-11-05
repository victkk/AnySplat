# DL3DV 评测对比方案

## 项目目标
在 DL3DV 数据集上对 AnySplat 和 SparseSplat 进行一致的评测对比，使用相同的 6 视角 context views 和 50 个 target views。

## 核心差异分析

### 1. 数据加载方式
| 项目 | 数据格式 | Dataset类型 | 加载方式 |
|------|---------|------------|---------|
| **SparseSplat** | `.torch` chunks | IterableDataset | 预处理的压缩格式 |
| **AnySplat** | 原始图像 + transforms.json | Dataset | 从原始文件加载 |

### 2. 评测索引系统
| 项目 | 索引方式 | 场景标识 | 可复现性 |
|------|---------|---------|---------|
| **SparseSplat** | 固定的 evaluation JSON | 场景哈希值 | ✅ 完全可复现 |
| **AnySplat** | LLFF holdout 动态采样 | 场景路径 | ❌ 不保证可复现 |

### 3. Evaluation JSON 格式
```json
{
  "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7": {
    "context": [0, 19, 29, 33, 39, 49],
    "target": [0, 1, 2, 3, ..., 49]
  }
}
```
- Key: 场景哈希值（SHA256）
- context: 6个固定的 context view 索引
- target: 50个 target view 索引（通常是0-49）

---

## 推荐方案：创建 AnySplat 的 DL3DV 评测适配器

### 方案优势
1. ✅ **最小化代码修改** - 只需添加新的评测脚本
2. ✅ **完全可复现** - 使用与 SparseSplat 相同的 view pairs
3. ✅ **利用现有基础设施** - 复用 AnySplat 的 dataloader 和 model
4. ✅ **独立性强** - 不影响原有代码

---

## 详细实施步骤

### 步骤 1: 场景哈希映射分析

**问题**: SparseSplat JSON 使用哈希值，AnySplat 使用路径

**解决方案选项**:

#### 选项 A: 分析哈希生成逻辑
1. 查看 SparseSplat 如何生成场景哈希
2. 找到哈希与场景路径/内容的对应关系
3. 在 AnySplat 中实现相同的哈希函数

**需要检查的文件**:
- `SparseSplat/src/scripts/generate_dl3dv_index.py`
- `SparseSplat/src/evaluation/evaluation_index_generator.py`
- DL3DV 原始数据结构

#### 选项 B: 创建映射文件
如果哈希是基于场景路径的 SHA256：
```python
import hashlib
scene_path = "test/scene_name"
scene_hash = hashlib.sha256(scene_path.encode()).hexdigest()
```

#### 选项 C: 手动映射（临时方案）
- 统计 evaluation JSON 中有多少个场景（140个test场景）
- 按顺序与 AnySplat 的 test_index.json 对应

---

### 步骤 2: 创建评测适配器脚本

**文件**: `AnySplat/src/eval_dl3dv.py`

#### 核心功能模块

##### 2.1 加载 evaluation indices
```python
def load_evaluation_indices(json_path: str) -> dict:
    """
    加载 SparseSplat 的 evaluation JSON

    Returns:
        dict: {scene_hash: {context: [...], target: [...]}}
    """
    with open(json_path, 'r') as f:
        return json.load(f)
```

##### 2.2 场景哈希映射
```python
def build_scene_hash_mapping(dataset_root: str, test_index_path: str) -> dict:
    """
    建立场景路径到哈希值的映射

    Returns:
        dict: {scene_hash: scene_path}
    """
    # 方法1: 如果哈希是基于路径
    mapping = {}
    with open(test_index_path, 'r') as f:
        scene_paths = json.load(f)

    for scene_path in scene_paths:
        # 需要确认哈希生成方式
        scene_hash = hashlib.sha256(scene_path.encode()).hexdigest()
        mapping[scene_hash] = scene_path

    return mapping
```

##### 2.3 数据加载适配
```python
def load_scene_with_indices(
    dataset: DatasetDL3DV,
    scene_id: str,
    context_indices: list,
    target_indices: list
) -> tuple:
    """
    从 AnySplat dataset 中加载指定索引的 views

    Args:
        dataset: AnySplat 的 DL3DV dataset
        scene_id: 场景ID
        context_indices: context view 索引列表
        target_indices: target view 索引列表

    Returns:
        context_images, target_images, context_cameras, target_cameras
    """
    scene_frames = dataset.scenes[scene_id]

    # 加载 context views
    context_data = [
        load_single_frame(dataset, scene_id, idx)
        for idx in context_indices
    ]

    # 加载 target views
    target_data = [
        load_single_frame(dataset, scene_id, idx)
        for idx in target_indices
    ]

    return process_data(context_data, target_data)
```

##### 2.4 推理和评测
```python
def evaluate_scene(
    model: AnySplat,
    context_images: torch.Tensor,
    target_images: torch.Tensor,
    target_extrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    device: torch.device
) -> dict:
    """
    对单个场景进行推理和评测

    Returns:
        dict: {psnr: float, ssim: float, lpips: float}
    """
    # 1. Encoder forward
    encoder_output = model.encoder(context_images, global_step=0)
    gaussians = encoder_output.gaussians

    # 2. 获取预测的 target poses
    # (参考 eval_nvs.py 的实现)
    pred_target_poses = predict_target_poses(
        model, context_images, target_images, encoder_output
    )

    # 3. Decoder forward
    output = model.decoder.forward(
        gaussians,
        pred_target_poses['extrinsic'],
        pred_target_poses['intrinsic'],
        near, far, image_shape
    )

    # 4. 计算指标
    psnr = compute_psnr(output.color, target_images)
    ssim = compute_ssim(output.color, target_images)
    lpips = compute_lpips(output.color, target_images)

    return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
```

##### 2.5 主评测循环
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                       help='DL3DV 数据根目录')
    parser.add_argument('--eval_json', type=str, required=True,
                       help='SparseSplat evaluation JSON 路径')
    parser.add_argument('--output_dir', type=str, default='outputs/dl3dv_eval')
    args = parser.parse_args()

    # 1. 加载模型
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda")
    model.to(device).eval()

    # 2. 加载 dataset
    dataset = create_dl3dv_dataset(args.data_root, stage='test')

    # 3. 加载 evaluation indices
    eval_indices = load_evaluation_indices(args.eval_json)

    # 4. 建立场景映射
    scene_mapping = build_scene_hash_mapping(args.data_root)

    # 5. 遍历所有场景
    all_metrics = []
    for scene_hash, indices in tqdm(eval_indices.items()):
        scene_path = scene_mapping[scene_hash]

        # 加载数据
        context_data, target_data = load_scene_with_indices(
            dataset, scene_path,
            indices['context'], indices['target']
        )

        # 评测
        metrics = evaluate_scene(model, context_data, target_data, device)
        all_metrics.append(metrics)

        # 保存结果
        save_results(args.output_dir, scene_hash, metrics)

    # 6. 汇总统计
    print_summary_statistics(all_metrics)
```

---

### 步骤 3: 数据预处理（可选）

如果 AnySplat 需要特定的数据预处理：

#### 3.1 相机参数归一化
SparseSplat 和 AnySplat 可能使用不同的相机参数格式：

```python
def normalize_camera_params(intrinsics, extrinsics, image_shape):
    """
    确保相机参数格式一致

    - SparseSplat: 可能使用归一化的 intrinsics
    - AnySplat: 检查其期望的格式
    """
    # 根据项目差异进行转换
    pass
```

#### 3.2 图像预处理
确保图像预处理一致：
- 归一化范围：[0, 1] vs [-1, 1]
- 分辨率：确保使用相同的图像分辨率
- 颜色空间：RGB vs BGR

---

### 步骤 4: 验证和调试

#### 4.1 单场景测试
先在单个场景上验证流程：
```bash
python src/eval_dl3dv.py \
  --data_root /path/to/dl3dv \
  --eval_json assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  --test_single_scene <scene_hash> \
  --output_dir outputs/debug
```

#### 4.2 检查清单
- [ ] 场景哈希映射正确
- [ ] 加载的 view indices 与 JSON 一致
- [ ] 图像分辨率正确（270x480 for images_4）
- [ ] 相机参数格式正确
- [ ] 渲染输出与 ground truth 对齐
- [ ] 指标计算无误

#### 4.3 对比验证
- 在 SparseSplat 上运行相同场景
- 对比输入数据是否一致
- 对比渲染结果的视觉效果

---

### 步骤 5: 完整评测

```bash
# AnySplat 评测
python AnySplat/src/eval_dl3dv.py \
  --data_root /path/to/dl3dv \
  --eval_json SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  --output_dir outputs/anysplat_dl3dv_6v

# SparseSplat 评测（参考）
cd SparseSplat
python -m src.main +experiment=dl3dv \
  dataset.roots=[/path/to/dl3dv] \
  dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
  mode=test \
  output_dir=outputs/sparsesplat_dl3dv_6v
```

---

## 替代方案对比

### 方案 2: 修改 SparseSplat 以支持 AnySplat 模型
**优点**: 利用 SparseSplat 成熟的评测基础设施
**缺点**:
- ❌ 需要大量修改 SparseSplat 代码
- ❌ 模型架构差异可能导致集成困难
- ❌ 维护成本高

### 方案 3: 转换数据格式
**优点**: 数据加载更快
**缺点**:
- ❌ 需要转换 DL3DV 数据为 .torch 格式
- ❌ 存储开销大
- ❌ 转换过程可能引入误差

---

## 关键技术细节

### 1. 场景哈希生成
需要确认 SparseSplat 如何生成场景哈希。可能的方式：
- SHA256(场景路径)
- SHA256(场景名称)
- 其他标识符

**检查方法**:
```python
# 在 SparseSplat 中查找
grep -r "hashlib" SparseSplat/src/
grep -r "sha256" SparseSplat/src/
```

### 2. 相机坐标系统
- **OpenCV**: +X right, +Y down, +Z forward
- **OpenGL/Blender**: +X right, +Y up, +Z backward

需要确认两个项目使用的坐标系统是否一致。

### 3. 深度范围
```python
# SparseSplat (from config)
near: 0.5
far: 200.0

# AnySplat (from eval_nvs.py)
near: 0.01
far: 100.0
```
可能需要统一深度范围。

### 4. 图像分辨率
```python
# SparseSplat config
image_shape: [270, 480]  # images_4 (4x downsampled)
# or
image_shape: [540, 960]  # images_8 (8x downsampled)

# AnySplat
# 需要检查其使用的分辨率
```

---

## 预期输出

### 评测指标
对于每个场景和整体数据集：
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity)
- **LPIPS** (Learned Perceptual Image Patch Similarity)

### 输出格式
```json
{
  "per_scene": {
    "scene_hash_1": {
      "psnr": 25.3,
      "ssim": 0.85,
      "lpips": 0.12
    },
    ...
  },
  "summary": {
    "psnr_mean": 24.5,
    "psnr_std": 2.1,
    "ssim_mean": 0.83,
    "ssim_std": 0.05,
    "lpips_mean": 0.15,
    "lpips_std": 0.03
  }
}
```

---

## 时间估算

假设 140 个测试场景：
- 单场景推理时间: ~5-10秒
- 总评测时间: ~15-30分钟（单GPU）

---

## 下一步行动

1. **立即执行**:
   - [ ] 分析场景哈希生成方式
   - [ ] 验证数据格式兼容性
   - [ ] 创建 eval_dl3dv.py 脚本框架

2. **短期目标**:
   - [ ] 在单个场景上验证流程
   - [ ] 完善数据加载逻辑
   - [ ] 调试并确保输出正确

3. **长期目标**:
   - [ ] 完整数据集评测
   - [ ] 结果分析和可视化
   - [ ] 撰写对比报告

---

## 参考文件清单

### SparseSplat
- `src/dataset/dataset_dl3dv.py` - DL3DV dataloader
- `src/dataset/view_sampler/view_sampler_evaluation.py` - Evaluation sampler
- `src/evaluation/evaluation_index_generator.py` - Index generator
- `assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json` - Evaluation indices

### AnySplat
- `src/dataset/dataset_dl3dv.py` - DL3DV dataloader
- `src/eval_nvs.py` - NVS evaluation script
- `src/model/model/anysplat.py` - Model definition
- `src/evaluation/metrics.py` - Metric computation

---

## 注意事项

1. **数据一致性**: 确保使用相同的图像预处理和相机参数
2. **可复现性**: 固定随机种子，使用确定性算法
3. **内存管理**: 140个场景可能需要批处理以避免OOM
4. **结果验证**: 与原论文报告的指标进行对比
5. **日志记录**: 详细记录每个场景的处理过程和潜在问题
