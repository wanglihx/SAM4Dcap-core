# TVB: Training-free Video to Biomechanics Twin System via Monocular Camera based on SAM3Dbody

## Paper
Paper in progress.

## What this project does
This project performs training-free biomechanics analysis from monocular video.


![2026-01-26T12_46_39 279Z-827938](https://github.com/user-attachments/assets/5310c0be-dbed-4304-bd30-52c63c1807cd) 

![2026-01-26T12_46_39 279Z-827938](https://github.com/user-attachments/assets/25ba24d9-c04c-4b7c-9e6e-71e06c910603)

## Quick demo
Monocular video results adapted to the opencap backend:

```bash
cd TVB/SAM4Dcap/output_viz
python -m http.server 8088 --bind 127.0.0.1
open http://127.0.0.1:8088/webviz_pipeline2/
```

## Prepare

### Hardware
- RTX PRO 6000 (96GB)
- 22 vCPU Intel(R) Xeon(R) Platinum 8470Q

### Environment
Environment paths and source projects
- TVB/envs/body4d: https://github.com/gaomingqi/sam-body4d
- TVB/envs/MHRtoSMPL: https://github.com/facebookresearch/MHR
- TVB/envs/opencap: https://github.com/opencap-org/opencap-core
- TVB/envs/opensim: https://github.com/opensim-org/opensim-core

We compiled with CUDA for the GPU architecture used in our experiments (sm_120) using these versions:
- MHRtoSMPL: Python 3.12.12; PyTorch 2.8.0+cu128; CUDA 12.8
- body4d: Python 3.12.12; PyTorch 2.8.0+cu128; CUDA 12.8
- opencap: Python 3.9.25; PyTorch 2.8.0+cu128; CUDA 12.8
- opensim: Python 3.10.19; PyTorch 2.9.1+cu128; CUDA 12.8

Full environment files will be uploaded to a cloud drive later.

### Codebases and models

#### We integrated six repositories
- https://github.com/gaomingqi/sam-body4d
- https://github.com/facebookresearch/MHR
- https://github.com/opencap-org/opencap-core
- https://github.com/opensim-org/opensim-core
- https://github.com/MarilynKeller/SMPL2AddBiomechanics
- https://github.com/keenon/AddBiomechanic

- Except for opensim （没修改）, all repositories have modified or new code. For details see: TVB/Readme_modified/README.md

修改的版本已经上传到本仓库的分支；对于Add需要从xxx下载, 并解压到add/fronted

#### Models
SMPL model download: https://smpl.is.tue.mpg.de/
Convert to the chumpy-free version with: 

`python TVB/MHRtoSMPL/convert_smpl_chumpy_free.py`

Model path checkpoints:
TVB/Readme_modified/checkpoints.txt

## One-click run + visualization

Double-check paths before running:
TVB/Readme_modified/check_again.txt

- Adapt AddBiomechanic with 105 keypoints:
```bash
bash pipeline1.sh
```
- Adapt opencap with 43 keypoints:
```bash
bash pipeline2.sh
```
- opencap reproduction:
```bash
bash opencap.sh
```
- Custom keypoints:
```bash
bash select.sh
```
- 对齐效果:

## Quick setup
Because of GitHub repository size limits, we will upload the complete project code and environments to a cloud drive. Contact wangli1@stu.scu.edu.cn to reproduce the project more easily.

## Next
We will further optimize pipeline1 and pipeline2 to achieve more accurate training-free IK solving and GRF analysis.

## Acknowledgements
Thanks to
- https://github.com/gaomingqi/sam-body4d
- https://github.com/facebookresearch/MHR
- https://github.com/opencap-org/opencap-core
- https://github.com/opensim-org/opensim-core
- https://github.com/MarilynKeller/SMPL2AddBiomechanics
- https://github.com/keenon/AddBiomechanic


