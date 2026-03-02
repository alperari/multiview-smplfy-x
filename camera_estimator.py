# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import json
import os
import os.path as osp
import shutil
import subprocess

import numpy as np


def _run_cmd(cmd, cwd=None):
    try:
        proc = subprocess.run(cmd, cwd=cwd, check=False,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True)
    except FileNotFoundError:
        raise RuntimeError(
            'Failed to run command: {}\n'
            'Executable not found: {}\n'
            'Install COLMAP and ensure it is on PATH, or pass '
            '--colmap_binary /absolute/path/to/colmap.'.format(
                ' '.join(cmd), cmd[0]))
    if proc.returncode != 0:
        raise RuntimeError(
            'Command failed with code {}: {}\n{}'.format(
                proc.returncode, ' '.join(cmd), proc.stdout))
    return proc.stdout


def _qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw,
         2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz,
         2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw,
         1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float32)


def _parse_cameras_txt(path):
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            cam_id = int(toks[0])
            model = toks[1]
            width = int(toks[2])
            height = int(toks[3])
            params = list(map(float, toks[4:]))
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params,
            }
    return cameras


def _fx_fy_cx_cy(model, width, height, params):
    if model == 'SIMPLE_PINHOLE':
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model == 'PINHOLE':
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    if model == 'SIMPLE_RADIAL':
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model == 'RADIAL':
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model == 'OPENCV':
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)

    raise ValueError('Unsupported COLMAP camera model: {}'.format(model))


def _parse_images_txt(path, cameras):
    entries = {}
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()
                 and not ln.startswith('#')]

    for idx in range(0, len(lines), 2):
        header = lines[idx].split()
        image_id = int(header[0])
        qw, qx, qy, qz = map(float, header[1:5])
        tx, ty, tz = map(float, header[5:8])
        cam_id = int(header[8])
        name = header[9]

        camera = cameras[cam_id]
        fx, fy, cx, cy = _fx_fy_cx_cy(
            camera['model'], camera['width'], camera['height'], camera['params'])

        stem = osp.splitext(osp.basename(name))[0]
        rot = _qvec_to_rotmat([qw, qx, qy, qz])
        trans = np.array([tx, ty, tz], dtype=np.float32)

        entries[stem] = {
            'image_id': image_id,
            'name': name,
            'cam_R': rot.tolist(),
            'cam_t': trans.tolist(),
            'cam_fx': fx,
            'cam_fy': fy,
            'cam_cx': cx,
            'cam_cy': cy,
            'width': camera['width'],
            'height': camera['height'],
        }

    return entries


def _count_registered(images_txt_path):
    if not osp.exists(images_txt_path):
        return 0
    with open(images_txt_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()
                 and not ln.startswith('#')]
    return len(lines) // 2


def _pick_best_sparse_model(sparse_root):
    if not osp.exists(sparse_root):
        raise RuntimeError('COLMAP sparse directory was not created: {}'.format(
            sparse_root))

    subdirs = [osp.join(sparse_root, d) for d in os.listdir(sparse_root)
               if osp.isdir(osp.join(sparse_root, d))]
    if not subdirs:
        raise RuntimeError(
            'COLMAP produced no sparse model in {}'.format(sparse_root))

    best = None
    best_count = -1
    for d in subdirs:
        images_txt = osp.join(d, 'images.txt')
        count = _count_registered(images_txt)
        if count > best_count:
            best_count = count
            best = d

    return best, best_count


def estimate_cameras_from_folder(image_folder,
                                 output_path,
                                 backend='colmap',
                                 overwrite=False,
                                 colmap_binary='colmap',
                                 colmap_work_dir='',
                                 colmap_matcher='exhaustive',
                                 colmap_camera_model='SIMPLE_PINHOLE',
                                 colmap_single_camera=True):
    backend = backend.lower()
    if backend != 'colmap':
        raise ValueError('Unsupported camera backend: {}'.format(backend))

    image_folder = osp.expandvars(image_folder)
    output_path = osp.expandvars(output_path)

    if not osp.exists(image_folder):
        raise ValueError(
            'Image folder does not exist: {}'.format(image_folder))

    if osp.exists(output_path) and (not overwrite):
        with open(output_path, 'r') as f:
            existing = json.load(f)
        return {
            'output_path': output_path,
            'num_cameras': len(existing.get('frames', {})),
            'used_cache': True,
            'colmap_work_dir': colmap_work_dir,
        }

    output_dir = osp.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not colmap_work_dir:
        colmap_work_dir = osp.join(output_dir if output_dir else '.', 'colmap')
    colmap_work_dir = osp.expandvars(colmap_work_dir)
    os.makedirs(colmap_work_dir, exist_ok=True)

    db_path = osp.join(colmap_work_dir, 'database.db')
    sparse_root = osp.join(colmap_work_dir, 'sparse')
    txt_root = osp.join(colmap_work_dir, 'sparse_txt')

    if osp.exists(db_path):
        os.remove(db_path)
    if osp.exists(sparse_root):
        shutil.rmtree(sparse_root)
    if osp.exists(txt_root):
        shutil.rmtree(txt_root)

    os.makedirs(sparse_root, exist_ok=True)

    _run_cmd([
        colmap_binary,
        'feature_extractor',
        '--database_path', db_path,
        '--image_path', image_folder,
        '--ImageReader.camera_model', str(colmap_camera_model),
        '--ImageReader.single_camera', '1' if colmap_single_camera else '0',
    ])

    if colmap_matcher == 'sequential':
        _run_cmd([
            colmap_binary,
            'sequential_matcher',
            '--database_path', db_path,
        ])
    else:
        _run_cmd([
            colmap_binary,
            'exhaustive_matcher',
            '--database_path', db_path,
        ])

    _run_cmd([
        colmap_binary,
        'mapper',
        '--database_path', db_path,
        '--image_path', image_folder,
        '--output_path', sparse_root,
    ])

    best_model_dir, num_registered = _pick_best_sparse_model(sparse_root)
    os.makedirs(txt_root, exist_ok=True)

    _run_cmd([
        colmap_binary,
        'model_converter',
        '--input_path', best_model_dir,
        '--output_path', txt_root,
        '--output_type', 'TXT',
    ])

    cameras_txt = osp.join(txt_root, 'cameras.txt')
    images_txt = osp.join(txt_root, 'images.txt')

    cameras = _parse_cameras_txt(cameras_txt)
    frames = _parse_images_txt(images_txt, cameras)

    out = {
        'backend': 'colmap',
        'image_folder': image_folder,
        'frames': frames,
    }
    with open(output_path, 'w') as f:
        json.dump(out, f)

    return {
        'output_path': output_path,
        'num_cameras': len(frames),
        'num_registered_images': int(num_registered),
        'used_cache': False,
        'colmap_work_dir': colmap_work_dir,
    }
