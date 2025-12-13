<!-- <h1 align="center">🏂 SAM-Body4D</h1> -->

# 🏂 SAM-Body4D

[**Mingqi Gao**](https://mingqigao.com), [**Yunqi Miao**](https://yoqim.github.io/), [**Jungong Han**](https://jungonghan.github.io/)

**SAM-Body4D** is a **training-free** method for **temporally consistent** and **robust** 4D human mesh recovery from videos.
By leveraging **pixel-level human continuity** from promptable video segmentation **together with occlusion recovery**, it reliably preserves identity and full-body geometry in challenging in-the-wild scenes.

[ 📄 [`Paper`](https://arxiv.org/pdf/2512.08406)] [ 🌐 [`Project Page`](https://mingqigao.com/projects/sam-body4d/index.html)] [ 📝 [`BibTeX`](#-citation)]


### ✨ Key Features

- **Temporally consistent human meshes across the entire video**
<div align=center>
<img src="./assets/demo1.gif" width="99%"/>
</div>

- **Robust multi-human recovery under heavy occlusions**
<div align=center>
<img src="./assets/demo2.gif" width="99%"/>
</div>

- **Robust 4D reconstruction under camera motion**
<div align=center>
<img src="./assets/demo3.gif" width="99%"/>
</div>

<!-- Training-Free 4D Human Mesh Recovery from Videos, based on [SAM-3](https://github.com/facebookresearch/sam3), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas), and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body). -->

## 🕹️ Gradio Demo

https://github.com/user-attachments/assets/07e49405-e471-40a0-b491-593d97a95465


## 📊 Resource & Profiling Summary

For detailed GPU/CPU resource usage, peak memory statistics, and runtime profiling, please refer to:

👉 **[resources.md](assets/doc/resources.md)**  


## 🖥️ Installation

#### 1. Create and Activate Environment
```
conda create -n body4d python=3.12 -y
conda activate body4d
```
#### 2. Install PyTorch (choose the version that matches your CUDA), Detectron, and SAM3
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install -e models/sam3
```
If you are using a different CUDA version, please select the matching PyTorch build from the official download page:
https://pytorch.org/get-started/previous-versions/

#### 3. Install Dependencies
```
pip install -e .
```


## 🚀 Run the Demo

#### 1. Setup checkpoints & config (recommended)

We provide an automated setup script that:
- generates `configs/body4d.yaml` from a release template,
- downloads all required checkpoints (existing files will be skipped).

Some checkpoints (**[SAM 3](https://huggingface.co/facebook/sam3)** and **[SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3)**) require prior access approval on Hugging Face.
Before running the setup script, please make sure you have **accepted access**
on their Hugging Face pages.

If you plan to use these checkpoints, login once:
```bash
huggingface-cli login
```
Then run the setup script:
```bash
python scripts/setup.py --force --ckpt-root /path/to/checkpoints
```
#### 2. Run
```bash
python app.py
```
#### Manual checkpoint setup (optional)

If you prefer to download checkpoints manually ([SAM 3](https://huggingface.co/facebook/sam3), [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3), [MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl-normal), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas?tab=readme-ov-file#download-checkpoints), [Depth-Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)), please place them under the directory with the following structure:
```
${CKPT_ROOT}/
├── sam3/                                
│   └── sam3.pt
├── sam-3d-body-dinov3/
│   ├── model.ckpt
│   └── assets/
│       └── mhr_model.pt
├── moge-2-vitl-normal/
│   └── model.pt
├── diffusion-vas-amodal-segmentation/
│   └── (directory contents)
├── diffusion-vas-content-completion/
│   └── (directory contents)
└── depth_anything_v2_vitl.pth
```
After placing the files correctly, you can run the setup script again.
Existing files will be detected and skipped automatically.

## 📝 Citation
If you find this repository useful, please consider giving a star ⭐ and citation.
```
@article{gao2025sambody4d,
  title   = {SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos},
  author  = {Gao, Mingqi and Miao, Yunqi and Han, Jungong},
  journal = {arXiv preprint arXiv:2512.08406},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.08406}
}
```

## 👏 Acknowledgements

The project is built upon [SAM-3](https://github.com/facebookresearch/sam3), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas) and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body). We sincerely thank the original authors for their outstanding work and contributions. 
