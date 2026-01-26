from pathlib import Path
import sys
import yaml

# add opencap-core to path
REPO_ROOT = Path(__file__).resolve().parents[2] / "opencap-core"
sys.path.append(str(REPO_ROOT))

from utilsOpenSim import getScaleTimeRange, runScaleTool  # noqa: E402

HERE = Path(__file__).parent

# inputs
STATIC_TRC = HERE / "static" / "static_43markers_aligned.trc"
SESSION_META = Path("/root/TVB/SAM4Dcap/opencap/data/subject2/Session0/sessionMetadata.yaml")
MODEL_NAME_FALLBACK = "LaiUhlrich2022"
SCALING_XML_UPRIGHT = REPO_ROOT / "opensimPipeline" / "Scaling" / "Setup_scaling_LaiUhlrich2022.xml"
SCALING_XML_ANYPOSE = REPO_ROOT / "opensimPipeline" / "Scaling" / "Setup_scaling_LaiUhlrich2022_any_pose.xml"
MODELS_DIR = REPO_ROOT / "opensimPipeline" / "Models"
OUT_DIR = HERE / "scaling_out"


def load_subject_params():
    if not SESSION_META.exists():
        raise FileNotFoundError(f"sessionMetadata not found: {SESSION_META}")
    meta = yaml.safe_load(SESSION_META.read_text())
    mass_kg = meta.get("mass_kg")
    height_m = meta.get("height_m")
    openSimModel = meta.get("openSimModel", MODEL_NAME_FALLBACK)
    scaling_setup = meta.get("scalingsetup", "upright_standing_pose")
    if mass_kg is None or height_m is None:
        raise ValueError("mass_kg or height_m missing in sessionMetadata.yaml")
    return mass_kg, height_m, openSimModel, scaling_setup


def main():
    if not STATIC_TRC.exists():
        raise FileNotFoundError(f"Static TRC not found: {STATIC_TRC}")

    mass_kg, height_m, openSimModel, scaling_setup = load_subject_params()
    scaling_xml = (
        SCALING_XML_ANYPOSE if scaling_setup == "any_pose" else SCALING_XML_UPRIGHT
    )
    generic_model = MODELS_DIR / f"{openSimModel}.osim"
    if not generic_model.exists():
        raise FileNotFoundError(f"Generic model not found: {generic_model}")

    OUT_DIR.mkdir(exist_ok=True)

    # Auto-detect the static window; loosen thresholds if detection fails
    time_range = getScaleTimeRange(
        str(STATIC_TRC),
        thresholdPosition=0.1,  # displacement threshold in meters (default 0.003)
        thresholdTime=0.05,      # minimum static duration in seconds (default 0.1)
        removeRoot=True,
    )

    scaled_model = runScaleTool(
        str(scaling_xml),
        str(generic_model),
        mass_kg,
        str(STATIC_TRC),
        time_range,
        str(OUT_DIR),
        subjectHeight=height_m,
        suffix_model="",
    )
    print("Scaled model written:", scaled_model)


if __name__ == "__main__":
    main()
