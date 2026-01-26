#!/usr/bin/env python3
"""
Export the updated mapping (manual picks + symmetry) to CSV.
Columns: smpl_name, smpl_vertex, opencap_name
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import yaml

ROOT = Path("/root/TVB/SAM4Dcap/select")
DATA_DIR = ROOT / "data"
PICKS_JSON = DATA_DIR / "manual_picks_new.json"
SMPL_MESH = ROOT / "viewer/smpl_frame0_aligned_mesh.json"
SYM_NPY = DATA_DIR / "sym_idxs.npy"
MAPPING_CSV = DATA_DIR / "mapping_43_bsm105_smpl24.csv"
BSM_YAML = DATA_DIR / "bsm_markers.yaml"
OUT_CSV = DATA_DIR / "updated_mapping.csv"

NOTE_TO_TARGET = {
    "rmeta5": "r_5meta_study",
    "rtoe": "r_toe_study",
    "rsh1repla": "r_sh1_study",
    "rsh2": "r_sh2_study",
    "rsh3repla": "r_sh3_study",
    "rthigh1": "r_thigh1_study",
    "rthigh2": "r_thigh2_study",
    "rthigh3": "r_thigh3_study",
}

# manual vertex overrides (when manual pick lacks vertex or needs override)
MANUAL_VID_OVERRIDE = {
    "rmeta5": 6718,  # hand-picked toe vertex
    "rtoe": 6703,    # RMT5
}


def main():
    picks = json.loads(PICKS_JSON.read_text()).get("picks", [])
    note_to_vid = {p["note"]: p["vertex"] for p in picks if p.get("vertex") is not None}
    sym = np.load(SYM_NPY)
    verts = np.array(json.loads(SMPL_MESH.read_text())["vertices"], float).reshape(-1, 3)
    bsm_index = yaml.safe_load(Path(BSM_YAML).read_text())
    bsm_index = {k: int(v) for k, v in bsm_index.items() if isinstance(v, (int, float))}

    # Start with original mapping (BSM->OpenCap)
    rows = []
    with MAPPING_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            bsm = r["BSM105"].strip()
            oc = r["OpenCap43"].strip()
            if not oc:
                continue
            if oc in NOTE_TO_TARGET.values():
                continue  # will replace
            if bsm in bsm_index:
                rows.append((f"{bsm}_smpl", bsm_index[bsm], f"{oc}"))

    # Add manual replacements (right + left, use note as smpl name)
    for note, target in NOTE_TO_TARGET.items():
        vid = MANUAL_VID_OVERRIDE.get(note, note_to_vid.get(note))
        if vid is None:
            continue
        rows.append((f"{note}_smpl", vid, target))
        lvid = int(sym[vid])
        ltarget = target.replace("r_", "L_")
        lnote = note.replace("r", "L", 1) if note.startswith("r") else f"L_{note}"
        rows.append((f"{lnote}_smpl", lvid, ltarget))

    # Dedup by opencap_name (keep last, i.e., manual)
    dedup = {}
    for smpl_name, vid, oc in rows:
        dedup[oc] = (smpl_name, vid, oc)
    final_rows = list(dedup.values())
    final_rows.sort(key=lambda x: x[2].lower())

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smpl_name", "smpl_vertex", "opencap_name"])
        writer.writerows(final_rows)
    print(f"[OK] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
