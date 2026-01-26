#!/usr/bin/env python3
"""
Export a static "scene" JSON from an OpenSim .osim model:
- Body frame translations/rotations in ground (default pose)
- Joint-derived skeleton edges (parent body -> child body)
- Marker positions in ground (filtered by suffix, default: *_study)

This is intended for lightweight web visualization (Three.js).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import opensim


def _base_physical_frame(frame: opensim.PhysicalFrame) -> opensim.PhysicalFrame:
    """Resolve PhysicalOffsetFrame -> underlying base PhysicalFrame (Body/ground)."""
    while frame.getConcreteClassName() == "PhysicalOffsetFrame":
        frame = opensim.PhysicalOffsetFrame.safeDownCast(frame).getParentFrame()
    return frame


def _vec3_to_list(v: opensim.Vec3) -> list[float]:
    arr = v.to_numpy()
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _euler_xyz_from_transform(xform: opensim.Transform) -> list[float]:
    # Match OpenCap VisualizerJsons: Body-fixed XYZ angles (radians).
    e = xform.R().convertRotationToBodyFixedXYZ().to_numpy()
    return [float(e[0]), float(e[1]), float(e[2])]


def export_scene(
    model_path: Path,
    marker_suffix: str,
    include_rotations: bool,
) -> dict[str, Any]:
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    model.realizePosition(state)

    bodies: dict[str, Any] = {}
    bodyset = model.getBodySet()
    for body in bodyset:
        xform = body.getTransformInGround(state)
        entry: dict[str, Any] = {
            "translation": _vec3_to_list(xform.T()),
        }
        if include_rotations:
            entry["rotation"] = _euler_xyz_from_transform(xform)
        bodies[body.getName()] = entry

    edges: list[list[str]] = []
    jointset = model.getJointSet()
    for joint in jointset:
        parent_frame = _base_physical_frame(joint.getParentFrame())
        child_frame = _base_physical_frame(joint.getChildFrame())
        parent_name = parent_frame.getName()
        child_name = child_frame.getName()
        if parent_name == "ground":
            continue
        edges.append([parent_name, child_name])

    markers: list[dict[str, Any]] = []
    markerset = model.getMarkerSet()
    for i in range(markerset.getSize()):
        marker = markerset.get(i)
        name = marker.getName()
        if marker_suffix and not name.endswith(marker_suffix):
            continue
        loc = marker.getLocationInGround(state)
        markers.append({"name": name, "position": _vec3_to_list(loc)})

    markers.sort(key=lambda m: m["name"])

    return {
        "meta": {
            "model_path": str(model_path),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "units": "m",
            "marker_suffix": marker_suffix,
            "include_rotations": include_rotations,
        },
        "bodies": bodies,
        "edges": edges,
        "markers": markers,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    default_model = root / "output/Data/subject2_Session0/OpenSimData/mmpose_0.8/2-cameras/Model/LaiUhlrich2022_generic.osim"
    default_out = Path(__file__).resolve().parent / "scene_LaiUhlrich2022_generic.json"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help="Path to .osim model (default: OpenCap LaiUhlrich2022_generic.osim).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--marker-suffix",
        type=str,
        default="_study",
        help="Only export markers whose names end with this suffix (default: _study).",
    )
    parser.add_argument(
        "--include-rotations",
        action="store_true",
        help="Also export per-body rotation (body-fixed XYZ, radians).",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    scene = export_scene(
        model_path=args.model,
        marker_suffix=args.marker_suffix,
        include_rotations=args.include_rotations,
    )
    args.out.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n")

    print(f"Wrote: {args.out}")
    print(f"Markers: {len(scene['markers'])}")
    print(f"Bodies:  {len(scene['bodies'])}")
    print(f"Edges:   {len(scene['edges'])}")


if __name__ == "__main__":
    main()
