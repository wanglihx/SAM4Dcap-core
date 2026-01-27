# TVB: Training-free Video to Biomechanics Twin System via Monocular Camera based on SAM3Dbody

## Paper
Paper in progress.

## What this project does
This project performs training-free biomechanics analysis from monocular video.


<table style="width: 100%;">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/85e59fb5-3e18-4963-96b7-f59c6b42601b" width="100%" />
      <br />Cam 1
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b493b0c1-44f8-4b76-b5ba-22631a100170" width="100%" />
      <br />Visualisation
    </td>
  </tr>
</table>


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

- Except for OpenSim (which remains unchanged), all repositories contain modified or new code. For detailed documentation, please refer to:
```bash
TVB/Readme_modified/README.md
```
- The modified versions have been uploaded to the branches of this repository. For Addbiomechanics, please download the file from https://figshare.com/articles/software/fronted/31150132?file=61359931 and extract the file to the

```bash
TVB/Addbiomechanics/fronted
```
  
#### Models
SMPL model download: https://smpl.is.tue.mpg.de/
Convert to the chumpy-free version with: 

`python TVB/MHRtoSMPL/convert_smpl_chumpy_free.py`

Model path checkpoints:
```bash
TVB/Readme_modified/checkpoints.txt
```

## One-click run + visualization

Double-check paths before running:
```bash
TVB/Readme_modified/check_again.txt
```
- Adapt AddBiomechanic with 105 keypoints (Monocular Video):
```bash
bash TVB/SAM4Dcap/pipeline1.sh
```

<table style="width: 100%;">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/c7a8dcde-5a8e-4d23-bf89-031994813491" width="500" />
      <br />local(Linux): http: //localhost:3088/
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/6399d47d-cdc1-4227-8599-76736a58dfea" width="500" />
      <br />online: https: //app.addbiomechanics.org/
    </td>
  </tr>
</table>

- Adapt opencap with 43 keypoints (Monocular Video):
```bash
bash TVB/SAM4Dcap/pipeline2.sh
```
- opencap reproduction (Binocular Video):
```bash
bash TVB/SAM4Dcap/opencap.sh
```
<table style="width: 100%;">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b493b0c1-44f8-4b76-b5ba-22631a100170" width="100%" />
      <br />Monocular Video: http://127.0.0.1:8093/webviz/
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/72f2379e-6c42-4784-8d9f-279a0c9fb662" width="100%"/>
      <br />Binocular Video: http://127.0.0.1:8090/web/webviz/
    </td>
  </tr>
</table>

- Tool for custom keypoints:
```bash
bash TVB/SAM4Dcap/select.sh
```

https://github.com/user-attachments/assets/853ba90e-bb06-4caa-b2e2-ef391232b23f


- Align:

```bash
cd TVB/SAM4Dcap/align/webviz_compare
python -m http.server 8092 --bind 127.0.0.1
```


https://github.com/user-attachments/assets/f84332d3-11d3-4629-8439-1e3975c51caa


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


