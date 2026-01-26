# MHR to SMPL 转换指南

将 Meta 的 MHR (Momentum Human Rig) 格式转换为标准 SMPL 格式。

## 环境要求

- Python 3.12
- PyTorch >= 2.8.0
- pymomentum-cpu
- smplx

### 创建环境

```bash
conda create -p /root/TVB/envs/MHRtoSMPL python=3.12 -y
conda activate /root/TVB/envs/MHRtoSMPL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pymomentum-cpu smplx tqdm numpy
```

## 文件准备

### 1. MHR 库
需要 MHR 官方库：`/root/TVB/MHR`

### 2. SMPL 模型
需要 chumpy-free 的 SMPL 模型文件。如果只有原始 pkl 文件（含 chumpy 对象），需要先转换：

```bash
# 使用有 chumpy 的环境运行
python convert_smpl_chumpy_free.py
```

这会生成 `SMPL_NEUTRAL_chumpy_free.pkl`

### 3. MHR 数据
MHR 参数文件目录，每帧一个 `*_data.npz` 文件，包含：
- `pred_vertices`: (18439, 3) MHR 顶点
- `focal_length`: 相机焦距
- `pred_cam_t`: 相机平移

## 转换步骤

### 运行转换

```bash
cd /root/TVB/MHR/tools/mhr_smpl_conversion
/root/TVB/envs/MHRtoSMPL/bin/python /root/TVB/MHRtoSMPL/convert_mhr_to_smpl.py
```

**重要**：必须在 `mhr_smpl_conversion` 目录下运行，因为代码使用相对路径加载资源文件。

### 输出

转换后的 SMPL 参数保存在 `/root/TVB/MHRtoSMPL/output/`，每帧一个 `.npz` 文件：

| 参数 | 形状 | 说明 |
|------|------|------|
| betas | (10,) | 身体形状参数 |
| body_pose | (69,) | 身体姿态 (23关节 × 3) |
| global_orient | (3,) | 全局朝向 |
| transl | (3,) | 平移 |
| vertices | (6890, 3) | SMPL 顶点 |
| fitting_error | () | 拟合误差 |

## 关键配置

### single_identity=True
同一人的视频序列应使用此选项，所有帧共享相同的 betas（身体形状），只优化姿态参数。

### 转换方法
MHR → SMPL 方向只支持 **PyTorch** 方法（不支持 PyMomentum）。每帧独立优化，无时序一致性。

## 注意事项

1. **导入顺序**：必须在 `import torch` 之前导入 `pymomentum`，否则会 segfault
   ```python
   import pymomentum.geometry
   import pymomentum.torch
   import torch  # 必须在 pymomentum 之后
   ```

2. **顶点数差异**：MHR 有 18439 个顶点，SMPL 只有 6890 个

3. **坐标系**：转换后的 SMPL 顶点在局部坐标系，需配合原始 `pred_cam_t` 使用才能正确渲染

4. **坐标单位修改（重要）**：需要修改 MHR 官方转换代码中的坐标单位处理

   **文件位置**：`/root/TVB/MHR/tools/mhr_smpl_conversion/conversion.py`

   **问题**：原始代码假设 MHR 数据使用厘米，会在 MHR→SMPL 转换时乘以 0.01（cm→m）。但 sam-body4d 输出的 MHR 数据已经是米为单位，无需转换。

   **修改位置**：第 461-467 行，注释掉坐标转换：
   ```python
   # Handle coordinate system conversion for input
   if direction == "mhr2smpl":
       # NOTE: Our MHR data is already in meters, so no conversion needed
       # Original code assumed MHR uses centimeters:
       # source_vertices = (
       #     source_vertices * ConversionConstants.CENTIMETERS_TO_METERS
       # )
       pass
   ```

   **不修改的后果**：转换后的 SMPL 顶点会缩小 100 倍，渲染时人体几乎不可见

## 文件结构

```
/root/TVB/MHRtoSMPL/
├── README.md                        # 本文档
├── convert_mhr_to_smpl.py          # 主转换脚本
├── convert_smpl_chumpy_free.py     # SMPL模型转换脚本
├── SMPL_NEUTRAL_chumpy_free.pkl    # 转换后的SMPL模型
└── output/                          # 转换结果
    ├── 00000000.npz
    ├── 00000001.npz
    └── ...
```
