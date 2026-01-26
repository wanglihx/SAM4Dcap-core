from pathlib import Path
import sys

# add opencap-core to path
REPO_ROOT = Path(__file__).resolve().parents[2] / "opencap-core"
sys.path.append(str(REPO_ROOT))

from utilsOpenSim import generateVisualizerJson  # noqa: E402

HERE = Path(__file__).parent
MODEL_NO_PATELLA = HERE / "scaling_out" / "LaiUhlrich2022_scaled_no_patella.osim"
IK_MOTION = HERE / "ik_out" / "subject2_43markers_aligned.mot"
VIS_DIR = HERE / "vis"
OUT_JSON = VIS_DIR / "subject2_43markers_aligned.json"


def main():
    if not MODEL_NO_PATELLA.exists():
        raise FileNotFoundError(f"Model without patella not found: {MODEL_NO_PATELLA}")
    if not IK_MOTION.exists():
        raise FileNotFoundError(f"IK motion not found: {IK_MOTION}")
    VIS_DIR.mkdir(exist_ok=True)
    generateVisualizerJson(
        str(MODEL_NO_PATELLA),
        str(IK_MOTION),
        str(OUT_JSON),
        vertical_offset=0.0,
        roundToRotations=4,
        roundToTranslations=4,
    )
    print("Visualizer JSON written:", OUT_JSON)


if __name__ == "__main__":
    main()
