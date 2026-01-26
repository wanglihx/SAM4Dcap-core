#!/usr/bin/env python3
"""
Rebuild scene_updated_mapping_overlay.json using updated_mapping.csv (manual picks + symmetry).
SMPL marker names use notes (e.g., rmeta5_smpl, rtoe_smpl), colors: yellow for manual,
red default for others. Opencap markers are untouched. Hips are kept from base scene.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import csv

ROOT = Path("/root/TVB/SAM4Dcap/select")
DATA_DIR = ROOT / "data"
VIEWER_DIR = ROOT / "viewer"
BASE_SCENE = VIEWER_DIR / "scene_subject2_static1_frame1_43markers_compare_with_adj.json"
SMPL_MESH = VIEWER_DIR / "smpl_frame0_aligned_mesh.json"
MAPPING_CSV = DATA_DIR / "updated_mapping.csv"
OUT_SCENE = VIEWER_DIR / "scene_updated_mapping_overlay.json"

# Notes used for manual picks (to color yellow)
MANUAL_NOTES = {"rmeta5_smpl", "rtoe_smpl", "rsh1repla_smpl", "rsh2_smpl", "rsh3repla_smpl", "rthigh1_smpl", "rthigh2_smpl", "rthigh3_smpl",
                "Lmeta5_smpl", "Ltoe_smpl", "Lsh1repla_smpl", "Lsh2_smpl", "Lsh3repla_smpl", "Lthigh1_smpl", "Lthigh2_smpl", "Lthigh3_smpl"}


def main():
    base = json.loads(BASE_SCENE.read_text())
    markers = base.get("markers", [])
    verts = np.array(json.loads(SMPL_MESH.read_text())["vertices"], float).reshape(-1, 3)

    # opencap markers untouched
    opencap_markers = [m for m in markers if m.get("name", "").endswith("_opencap")]

    # hips from base
    hip_markers = []
    for m in markers:
        name = m.get("name", "")
        if name in ("RHJC_study_smpl", "LHJC_study_smpl"):
            hip_markers.append(m)

    # load mapping csv
    rows = []
    with MAPPING_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["smpl_name"], int(r["smpl_vertex"]), r["opencap_name"]))

    smpl_markers = []
    for smpl_name, vid, oc in rows:
        pos = verts[vid]
        color = [1, 1, 0] if smpl_name in MANUAL_NOTES else None
        m = {"name": smpl_name, "position": pos.tolist()}
        if color:
            m["color"] = color
        smpl_markers.append(m)

    new_markers = opencap_markers + hip_markers + smpl_markers

    out_scene = {
        "meta": {
            "note": "Scene rebuilt from updated_mapping.csv; manual points in yellow.",
            "base_scene": str(BASE_SCENE.resolve()),
            "mapping_csv": str(MAPPING_CSV.resolve()),
        },
        "bodies": base.get("bodies", {}),
        "edges": base.get("edges", []),
        "markers": new_markers,
    }
    OUT_SCENE.write_text(json.dumps(out_scene, indent=2))
    print(f"[OK] wrote {OUT_SCENE}")


if __name__ == "__main__":
    main()
