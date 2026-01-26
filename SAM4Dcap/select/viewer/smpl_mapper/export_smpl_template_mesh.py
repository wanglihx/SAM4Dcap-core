#!/usr/bin/env python3
"""
Export SMPL template mesh to a browser-friendly JSON for 0108addbio/smpl_mapper.

Input: SMPL pkl (dict) with at least:
  - v_template: (6890,3) float
  - f: (13776,3) int

Output JSON (compact, flat arrays):
  {
    "meta": {...},
    "vertices": [x0,y0,z0,x1,y1,z1,...],
    "faces": [a0,b0,c0,a1,b1,c1,...]
  }
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
import csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smpl-pkl",
        default="/root/TVB/MHRtoSMPL/SMPL_NEUTRAL_chumpy_free.pkl",
        help="SMPL neutral pkl (chumpy-free) path",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("smpl_template_mesh.json")),
        help="Output JSON path",
    )
    args = parser.parse_args()

    try:
        import numpy as np  # noqa: F401
    except Exception as e:
        print(f"ERROR: numpy is required to export: {e}", file=sys.stderr)
        return 2

    try:
        import scipy  # noqa: F401
    except Exception:
        print(
            "ERROR: scipy is required to unpickle SMPL (pkl contains scipy.sparse objects).\n"
            "Please run in the opencap environment, for example:\n"
            "  source /root/miniconda3/etc/profile.d/conda.sh\n"
            "  conda activate /root/TVB/envs/opencap\n"
            f"  python {Path(__file__).as_posix()}",
            file=sys.stderr,
        )
        return 2

    import numpy as np

    smpl_pkl = Path(args.smpl_pkl)
    if not smpl_pkl.exists():
        print(f"ERROR: SMPL pkl not found: {smpl_pkl}", file=sys.stderr)
        return 2

    print(f"Loading SMPL pkl: {smpl_pkl}")
    with smpl_pkl.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    if not isinstance(data, dict):
        print(f"ERROR: pkl top-level is not dict: {type(data)}", file=sys.stderr)
        return 2

    if "v_template" not in data or "f" not in data:
        keys = ", ".join(sorted(map(str, data.keys())))
        print(f"ERROR: pkl missing v_template or f; keys={keys}", file=sys.stderr)
        return 2

    v = np.asarray(data["v_template"], dtype=np.float32)
    f_arr = np.asarray(data["f"], dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3:
        print(f"ERROR: invalid v_template shape: {v.shape}", file=sys.stderr)
        return 2
    if f_arr.ndim != 2 or f_arr.shape[1] != 3:
        print(f"ERROR: invalid f shape: {f_arr.shape}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "meta": {
            "source_pkl": smpl_pkl.as_posix(),
            "vertex_count": int(v.shape[0]),
            "face_count": int(f_arr.shape[0]),
            "units": "unknown",
            "note": "Flat arrays. vertices length = vertex_count*3, faces length = face_count*3.",
        },
        "vertices": v.reshape(-1).tolist(),
        "faces": f_arr.reshape(-1).tolist(),
    }

    # Provide a small set of suggested anchor vertices for auto-alignment in the web UI.
    # These are heuristics on the neutral template mesh, primarily for quick overlay.
    try:
        import numpy as np

        anchors = {}
        Jreg = data.get("J_regressor")
        if Jreg is not None:
            J = np.asarray(Jreg.dot(v.astype(np.float64)), dtype=np.float64)

            def pick_lateral_knee_vertex(joint_xyz: np.ndarray, side: str) -> int:
                joint_xyz = joint_xyz.reshape(1, 3)
                d = np.linalg.norm(v.astype(np.float64) - joint_xyz, axis=1)
                cand = np.where(d < 0.12)[0]
                if cand.size < 10:
                    cand = np.where(np.abs(v[:, 1].astype(np.float64) - float(joint_xyz[0, 1])) < 0.06)[0]
                if cand.size == 0:
                    return int(np.argmin(d))
                if side.upper().startswith("L"):
                    return int(cand[np.argmax(v[cand, 0])])
                return int(cand[np.argmin(v[cand, 0])])

            def pick_c7_vertex(neck_xyz: np.ndarray) -> int:
                neck_xyz = neck_xyz.reshape(1, 3)
                d = np.linalg.norm(v.astype(np.float64) - neck_xyz, axis=1)
                cand = np.where(d < 0.12)[0]
                if cand.size == 0:
                    return int(np.argmin(d))
                cand2 = cand[np.where(np.abs(v[cand, 0].astype(np.float64)) < 0.05)[0]]
                use = cand2 if cand2.size else cand
                # C7 is posterior; assume back is smaller z on template.
                return int(use[np.argmin(v[use, 2])])

            # SMPL joint indices (SMPL 24):
            # 4/5 = knees, 12 = neck (used as C7 proxy)
            vid_lknee = pick_lateral_knee_vertex(J[4], "L")
            vid_rknee = pick_lateral_knee_vertex(J[5], "R")
            vid_c7 = pick_c7_vertex(J[12])

            anchors = {
                "C7_study": {"vertex": int(vid_c7)},
                "L_knee_study": {"vertex": int(vid_lknee)},
                "r_knee_study": {"vertex": int(vid_rknee)},
            }

            out["meta"]["autofit_anchors"] = anchors
            out["meta"]["autofit_note"] = "Heuristic SMPL template anchor vertices for quick alignment (C7 + bilateral knees)."

            # Heights: C7-to-ground (ground = min Y of template mesh).
            min_vy = float(np.min(v[:, 1].astype(np.float64)))
            smpl_c7_ground_m = float(v[int(vid_c7), 1].astype(np.float64)) - min_vy
            out["meta"]["smpl_c7_ground_m"] = smpl_c7_ground_m

            # Optional: compute BSM C7-to-ground from local CSV (ground = min Y of markers).
            bsm_csv = Path("/root/bsm_105_tpose.csv")
            if bsm_csv.exists():
                min_y = 1e9
                c7_y = None
                with bsm_csv.open("r", newline="") as fcsv:
                    reader = csv.DictReader(fcsv)
                    for row in reader:
                        y = float(row["y"])
                        min_y = min(min_y, y)
                        if row.get("marker_name") == "C7":
                            c7_y = y
                if c7_y is not None:
                    bsm_c7_ground_m = float(c7_y - min_y)
                    out["meta"]["bsm_c7_ground_m"] = bsm_c7_ground_m
                    out["meta"]["bsm_c7_ground_source"] = bsm_csv.as_posix()
                    if smpl_c7_ground_m > 1e-9:
                        out["meta"]["scale_to_bsm_c7_ground"] = float(bsm_c7_ground_m / smpl_c7_ground_m)
                        out["meta"]["scale_note"] = "scale_to_bsm_c7_ground makes SMPL C7-ground match BSM (both using min-Y as ground)."
    except Exception as e:
        out["meta"]["autofit_error"] = f"{type(e).__name__}: {e}"

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    print(f"Writing: {out_path} (tmp={tmp.name})")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done. {out_path} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
