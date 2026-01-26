#!/usr/bin/env python3
"""
Create a one-frame OpenSim scene JSON from:
- A VisualizerJsons static trial (for body transforms/scale/meshes)
- A TRC file (for markers)

The output matches `scene_LaiUhlrich2022_generic.json` schema so it can be
viewed directly in the static web viewer under 0108addbio/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List


def _flatten3(value):
    """Ensure [x, y, z] even if source is [[x, y, z]]."""
    if value is None:
        return [0.0, 0.0, 0.0]
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return value


def load_static_bodies(static_json: Path) -> Dict[str, dict]:
    data = json.loads(static_json.read_text())
    bodies = {}
    for name, body in data.get("bodies", {}).items():
        out = {
            "translation": _flatten3(body.get("translation")),
            "rotation": _flatten3(body.get("rotation")),
        }
        if "attachedGeometries" in body:
            out["attachedGeometries"] = body["attachedGeometries"]
        if "scaleFactors" in body:
            out["scaleFactors"] = body["scaleFactors"]
        bodies[name] = out
    return bodies


def parse_trc_frame(trc_path: Path, frame_number: int) -> Dict[str, List[float]]:
    """
    Parse a TRC and return marker -> [x, y, z] for the specified frame (1-based).
    """
    lines = [ln.strip() for ln in trc_path.read_text().splitlines() if ln.strip()]
    try:
        header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Frame#"))
    except StopIteration:
        raise RuntimeError(f"TRC missing 'Frame#' header: {trc_path}")

    marker_names = lines[header_idx].split()[2:]  # drop Frame# Time

    # Data lines start after the axis header (next line)
    data_lines = lines[header_idx + 2 :]
    target = None
    for ln in data_lines:
        parts = ln.split()
        if not parts:
            continue
        try:
            frame_id = int(float(parts[0]))
        except ValueError:
            continue
        if frame_id == frame_number:
            target = parts
            break

    if target is None:
        raise RuntimeError(f"Frame {frame_number} not found in {trc_path}")

    expected = 2 + 3 * len(marker_names)
    if len(target) < expected:
        raise RuntimeError(
            f"Unexpected column count in {trc_path} frame {frame_number}: "
            f"got {len(target)}, expected >= {expected}"
        )

    coords: Dict[str, List[float]] = {}
    for idx, name in enumerate(marker_names):
        base = 2 + idx * 3
        x, y, z = map(float, target[base : base + 3])
        coords[name] = [x, y, z]
    return coords


def load_edges(edges_scene: Path) -> List[List[str]]:
    data = json.loads(edges_scene.read_text())
    return data.get("edges", [])


def main() -> None:
    here = Path(__file__).resolve().parent
    default_static = (
        here.parent
        / "1230opencap"
        / "Data"
        / "subject2_Session0"
        / "VisualizerJsons"
        / "static1"
        / "static1.json"
    )
    default_trc = (
        here.parent
        / "1230opencap"
        / "Data"
        / "subject2_Session0"
        / "MarkerData"
        / "mmpose_0.8"
        / "2-cameras"
        / "PostAugmentation_v0.2"
        / "static1_LSTM.trc"
    )
    default_edges = here / "scene_LaiUhlrich2022_generic.json"
    default_model = (
        here.parent
        / "1230opencap"
        / "Data"
        / "subject2_Session0"
        / "OpenSimData"
        / "mmpose_0.8"
        / "2-cameras"
        / "Model"
        / "LaiUhlrich2022_scaled.osim"
    )
    default_out = here / "scene_subject2_static1_frame1.json"

    parser = argparse.ArgumentParser(
        description="Export one-frame scene JSON for 0108addbio viewer."
    )
    parser.add_argument("--static-json", type=Path, default=default_static, help="VisualizerJsons static JSON path.")
    parser.add_argument("--trc", type=Path, default=default_trc, help="TRC with markers (PostAugmentation).")
    parser.add_argument("--frame", type=int, default=1, help="1-based frame number to export (default: 1).")
    parser.add_argument(
        "--edges-scene",
        type=Path,
        default=default_edges,
        help="Scene JSON to copy edges from (default: scene_LaiUhlrich2022_generic.json).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help="OpenSim model path to record in meta (default: scaled subject2 model).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output scene JSON path (default: scene_subject2_static1_frame1.json in 0108addbio).",
    )
    args = parser.parse_args()

    bodies = load_static_bodies(args.static_json)
    coords = parse_trc_frame(args.trc, args.frame)
    markers = [{"name": n, "position": coords[n]} for n in sorted(coords) if n.endswith("_study")]
    edges = load_edges(args.edges_scene)

    meta = {
        "model_path": str(args.model.resolve()),
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "units": "m",
        "marker_suffix": "_study",
        "include_rotations": True,
        "include_meshes": True,
        "source_static_json": str(args.static_json.resolve()),
        "source_trc": str(args.trc.resolve()),
        "frame_number": args.frame,
    }

    output = {
        "meta": meta,
        "bodies": bodies,
        "edges": edges,
        "markers": markers,
    }

    args.out.write_text(json.dumps(output, indent=2))
    print(
        f"[OK] wrote {args.out}  markers={len(markers)} bodies={len(bodies)} frame={args.frame} "
        f"(model={args.model})"
    )


if __name__ == "__main__":
    main()
