from pathlib import Path
import sys

# add opencap-core to path
REPO_ROOT = Path(__file__).resolve().parents[2] / "opencap-core"
sys.path.append(str(REPO_ROOT))

from utilsOpenSim import runIKTool  # noqa: E402

HERE = Path(__file__).parent

SCALED_MODEL = HERE / "scaling_out" / "LaiUhlrich2022_scaled.osim"
DYNAMIC_TRC = HERE / "motion" / "subject2_43markers_aligned.trc"
IK_SETUP = REPO_ROOT / "opensimPipeline" / "IK" / "Setup_IK.xml"
IK_OUT_DIR = HERE / "ik_out"


def main():
    if not SCALED_MODEL.exists():
        raise FileNotFoundError(f"Scaled model not found: {SCALED_MODEL}")
    if not DYNAMIC_TRC.exists():
        raise FileNotFoundError(f"Dynamic TRC not found: {DYNAMIC_TRC}")
    if not IK_SETUP.exists():
        raise FileNotFoundError(f"IK setup not found: {IK_SETUP}")

    IK_OUT_DIR.mkdir(exist_ok=True)
    motion_path, model_no_patella = runIKTool(
        str(IK_SETUP),
        str(SCALED_MODEL),
        str(DYNAMIC_TRC),
        str(IK_OUT_DIR),
        IKFileName=DYNAMIC_TRC.stem,
    )
    print("IK motion written:", motion_path)
    print("Model without patella:", model_no_patella)


if __name__ == "__main__":
    main()
