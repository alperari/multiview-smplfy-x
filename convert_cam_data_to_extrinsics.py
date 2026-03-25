#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Encountered near-zero camera axis vector.")
    return v / n


def _to_float3(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"Expected length-3 vector, got shape {arr.shape}.")
    return arr


def build_extrinsics(cam_struct, opencv_axes=False):
    center = _to_float3(cam_struct.center)
    right = _normalize(_to_float3(cam_struct.right))
    up = _normalize(_to_float3(cam_struct.up))
    direction = _normalize(_to_float3(cam_struct.direction))

    # Many render datasets define +Y as up, while OpenCV images use +Y down.
    if opencv_axes:
        up = -up

    # Camera-to-world basis in columns, then invert by transpose.
    r_c2w = np.column_stack([right, up, direction])
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ center

    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = r_w2c
    ext[:3, 3] = t_w2c
    return ext, center, right, up, direction


def _field_names(cam_struct):
    names = getattr(cam_struct, "_fieldnames", None)
    if names is not None:
        return set(names)
    return {k for k in cam_struct.__dict__.keys() if not k.startswith("_")}


def _has_field(cam_struct, name):
    return name in _field_names(cam_struct)


def _extract_k_from_cam(cam_struct):
    for key in ("K", "intrinsic", "intrinsics", "A"):
        if _has_field(cam_struct, key):
            k = np.asarray(getattr(cam_struct, key), dtype=np.float64).reshape(3, 3)
            return k
    return None


def _extract_intrinsics_from_cam(cam_struct):
    k = _extract_k_from_cam(cam_struct)
    if k is not None:
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    names = _field_names(cam_struct)
    if {"fx", "fy", "cx", "cy"}.issubset(names):
        return (
            float(np.asarray(getattr(cam_struct, "fx")).reshape(-1)[0]),
            float(np.asarray(getattr(cam_struct, "fy")).reshape(-1)[0]),
            float(np.asarray(getattr(cam_struct, "cx")).reshape(-1)[0]),
            float(np.asarray(getattr(cam_struct, "cy")).reshape(-1)[0]),
        )
    return None


def _read_png_size(path):
    with path.open("rb") as f:
        data = f.read(24)
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    return w, h


def _read_jpeg_size(path):
    with path.open("rb") as f:
        data = f.read()
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    n = len(data)
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF
    }
    while i + 9 < n:
        while i < n and data[i] != 0xFF:
            i += 1
        if i + 1 >= n:
            return None
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = data[i]
        i += 1
        if marker in (0xD8, 0xD9):
            continue
        if i + 1 >= n:
            return None
        seg_len = (data[i] << 8) + data[i + 1]
        if seg_len < 2 or i + seg_len > n:
            return None
        if marker in sof_markers:
            if i + 6 >= n:
                return None
            h = (data[i + 3] << 8) + data[i + 4]
            w = (data[i + 5] << 8) + data[i + 6]
            return w, h
        i += seg_len
    return None


def _image_size(path):
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _read_png_size(path)
    if suffix in (".jpg", ".jpeg"):
        return _read_jpeg_size(path)
    return None


def _infer_size_from_folder(folder):
    if folder is None or not folder.exists():
        return None
    exts = (".jpg", ".jpeg", ".png")
    images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])
    for p in images:
        size = _image_size(p)
        if size is not None:
            return size
    return None


def _resolve_intrinsics(cams, num_cams, args):
    if any(v is not None for v in (args.fx, args.fy, args.cx, args.cy)):
        if any(v is None for v in (args.fx, args.fy, args.cx, args.cy)):
            raise ValueError("If overriding intrinsics, provide all of --fx --fy --cx --cy.")
        fxs = np.full((num_cams,), float(args.fx), dtype=np.float64)
        fys = np.full((num_cams,), float(args.fy), dtype=np.float64)
        cxs = np.full((num_cams,), float(args.cx), dtype=np.float64)
        cys = np.full((num_cams,), float(args.cy), dtype=np.float64)
        return fxs, fys, cxs, cys, "cli_override"

    # 1) Try reading per-camera intrinsics directly from cam_data.mat struct fields.
    values = []
    for cam in cams:
        val = _extract_intrinsics_from_cam(cam)
        if val is None:
            values = []
            break
        values.append(val)
    if values:
        arr = np.asarray(values, dtype=np.float64)
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], "cam_data_mat"

    # 2) Fallback used by this codebase: focal=5000, principal point at image center.
    input_mat = Path(args.input_mat)
    data_root = input_mat.parent.parent
    if args.image_folder:
        image_folder = Path(args.image_folder)
    else:
        image_folder = data_root / "color"
    size = _infer_size_from_folder(image_folder)
    if size is None:
        cx, cy = 0.0, 0.0
        source = "default_focal_no_image_size"
    else:
        w, h = size
        cx, cy = w / 2.0, h / 2.0
        source = "default_focal_image_center"

    fxs = np.full((num_cams,), float(args.default_focal), dtype=np.float64)
    fys = np.full((num_cams,), float(args.default_focal), dtype=np.float64)
    cxs = np.full((num_cams,), float(cx), dtype=np.float64)
    cys = np.full((num_cams,), float(cy), dtype=np.float64)
    return fxs, fys, cxs, cys, source


def main():
    parser = argparse.ArgumentParser(
        description="Convert cam_data.mat camera structs into per-camera world-to-camera extrinsic matrices."
    )
    parser.add_argument(
        "--input_mat",
        default="dataset_example/image_data/rp_dennis_posed_004/meta/cam_data.mat",
        help="Path to cam_data.mat",
    )
    parser.add_argument(
        "--output_npz",
        default="dataset_example/image_data/rp_dennis_posed_004/meta/cam_extrinsics.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Optional JSON output path (human-readable).",
    )
    parser.add_argument(
        "--output_txt",
        default="",
        help="Optional TXT output path in rig format: id, K, k1 k2, and 3x4 extrinsics.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Focal length x for TXT export.")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y for TXT export.")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x for TXT export.")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y for TXT export.")
    parser.add_argument(
        "--image_folder",
        default="",
        help="Optional image folder used to infer principal point (defaults to sibling 'color').",
    )
    parser.add_argument(
        "--default_focal",
        type=float,
        default=5000.0,
        help="Fallback focal length when cam_data.mat has no intrinsics (matches data_parser default).",
    )
    parser.add_argument("--k1", type=float, default=0.0, help="Radial distortion k1 for TXT export.")
    parser.add_argument("--k2", type=float, default=0.0, help="Radial distortion k2 for TXT export.")
    parser.add_argument(
        "--opencv_axes",
        action="store_true",
        help="Flip camera up axis to match OpenCV (+x right, +y down, +z forward).",
    )
    args = parser.parse_args()

    mat = loadmat(args.input_mat, squeeze_me=True, struct_as_record=False)
    if "cam" not in mat:
        raise KeyError(f"'cam' variable not found in {args.input_mat}")

    cams = np.atleast_1d(mat["cam"]).reshape(-1)
    num_cams = cams.shape[0]
    fxs, fys, cxs, cys, intr_source = _resolve_intrinsics(cams, num_cams, args)

    extrinsics = []
    centers = []
    rights = []
    ups = []
    directions = []
    for cam in cams:
        ext, c, r, u, d = build_extrinsics(cam, opencv_axes=args.opencv_axes)
        extrinsics.append(ext)
        centers.append(c)
        rights.append(r)
        ups.append(u)
        directions.append(d)

    extrinsics = np.stack(extrinsics, axis=0)
    centers = np.stack(centers, axis=0)
    rights = np.stack(rights, axis=0)
    ups = np.stack(ups, axis=0)
    directions = np.stack(directions, axis=0)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        extrinsics_w2c=extrinsics,
        centers_world=centers,
        right_world=rights,
        up_world=ups,
        direction_world=directions,
        opencv_axes=np.array([args.opencv_axes], dtype=np.bool_),
    )

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "opencv_axes": bool(args.opencv_axes),
            "num_cameras": int(extrinsics.shape[0]),
            "extrinsics_w2c": extrinsics.tolist(),
        }
        output_json.write_text(json.dumps(payload, indent=2))

    if args.output_txt:
        output_txt = Path(args.output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)

        def fmt(x):
            x = float(x)
            if abs(x) < 1e-12:
                x = 0.0
            s = f"{x:.7f}".rstrip("0").rstrip(".")
            if "." not in s:
                s += ".0"
            return s

        lines = []
        for cam_idx in range(extrinsics.shape[0]):
            lines.append(str(cam_idx))
            lines.append(f"{fmt(fxs[cam_idx])} 0.0 {fmt(cxs[cam_idx])}")
            lines.append(f"0.0 {fmt(fys[cam_idx])} {fmt(cys[cam_idx])}")
            lines.append("0.0 0.0 1.0")
            lines.append(f"{fmt(args.k1)} {fmt(args.k2)}")
            ext = extrinsics[cam_idx, :3, :]
            for r in range(3):
                lines.append(
                    f"{fmt(ext[r, 0])} {fmt(ext[r, 1])} {fmt(ext[r, 2])} {fmt(ext[r, 3])}"
                )
            lines.append("")
        output_txt.write_text("\n".join(lines).rstrip() + "\n")

    print(f"Saved {extrinsics.shape[0]} extrinsics to: {output_npz}")
    if args.output_json:
        print(f"Saved JSON to: {args.output_json}")
    if args.output_txt:
        print(f"Saved TXT to: {args.output_txt}")
        print(f"TXT intrinsics source: {intr_source}")


if __name__ == "__main__":
    main()
