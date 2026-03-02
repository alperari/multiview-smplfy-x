# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import json

import cv2
import numpy as np


COCO25_NUM_JOINTS = 25


def _pose33_to_coco25(landmarks, width, height):
    keypoints = np.zeros((COCO25_NUM_JOINTS, 3), dtype=np.float32)

    def get_xyv(idx):
        lm = landmarks[idx]
        x = float(lm.x) * width
        y = float(lm.y) * height
        v = float(np.clip(getattr(lm, 'visibility', 0.0), 0.0, 1.0))
        return np.array([x, y, v], dtype=np.float32)

    # Core body
    keypoints[0] = get_xyv(0)    # Nose
    keypoints[2] = get_xyv(12)   # RShoulder
    keypoints[3] = get_xyv(14)   # RElbow
    keypoints[4] = get_xyv(16)   # RWrist
    keypoints[5] = get_xyv(11)   # LShoulder
    keypoints[6] = get_xyv(13)   # LElbow
    keypoints[7] = get_xyv(15)   # LWrist
    keypoints[9] = get_xyv(24)   # RHip
    keypoints[10] = get_xyv(26)  # RKnee
    keypoints[11] = get_xyv(28)  # RAnkle
    keypoints[12] = get_xyv(23)  # LHip
    keypoints[13] = get_xyv(25)  # LKnee
    keypoints[14] = get_xyv(27)  # LAnkle

    # Face sparse
    keypoints[15] = get_xyv(5)   # REye
    keypoints[16] = get_xyv(2)   # LEye
    keypoints[17] = get_xyv(8)   # REar
    keypoints[18] = get_xyv(7)   # LEar

    # Feet
    keypoints[19] = get_xyv(31)  # LBigToe
    keypoints[20] = get_xyv(31)  # LSmallToe (approx)
    keypoints[21] = get_xyv(29)  # LHeel
    keypoints[22] = get_xyv(32)  # RBigToe
    keypoints[23] = get_xyv(32)  # RSmallToe (approx)
    keypoints[24] = get_xyv(30)  # RHeel

    # Neck and MidHip from averages
    l_sh = keypoints[5]
    r_sh = keypoints[2]
    l_hip = keypoints[12]
    r_hip = keypoints[9]

    keypoints[1, :2] = 0.5 * (l_sh[:2] + r_sh[:2])
    keypoints[1, 2] = min(l_sh[2], r_sh[2])

    keypoints[8, :2] = 0.5 * (l_hip[:2] + r_hip[:2])
    keypoints[8, 2] = min(l_hip[2], r_hip[2])

    return keypoints


def _empty_openpose_person():
    return {
        'pose_keypoints_2d': [0.0] * (COCO25_NUM_JOINTS * 3),
        'face_keypoints_2d': [],
        'hand_left_keypoints_2d': [],
        'hand_right_keypoints_2d': []
    }


def _to_openpose_json_dict(coco25_keypoints):
    person = _empty_openpose_person()
    person['pose_keypoints_2d'] = coco25_keypoints.reshape(-1).tolist()
    return {
        'version': 1.3,
        'people': [person],
    }


def extract_keypoints_from_folder(image_folder,
                                  output_folder,
                                  backend='mediapipe',
                                  overwrite=False,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5):
    backend = backend.lower()
    if backend != 'mediapipe':
        raise ValueError('Unsupported keypoint backend: {}'.format(backend))

    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            'mediapipe is required for automatic keypoint extraction. '
            'Install with: pip install mediapipe')

    image_folder = osp.expandvars(image_folder)
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(image_folder):
        raise ValueError('Image folder does not exist: {}'.format(image_folder))
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_paths = [osp.join(image_folder, fn)
                   for fn in os.listdir(image_folder)
                   if (fn.lower().endswith(valid_exts) and not fn.startswith('.'))]
    image_paths = sorted(image_paths)

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )

    total = 0
    detected = 0
    skipped = 0

    for img_path in image_paths:
        total += 1
        img_fn = osp.splitext(osp.basename(img_path))[0]
        out_path = osp.join(output_folder, img_fn + '_keypoints.json')

        if (not overwrite) and osp.exists(out_path):
            skipped += 1
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            openpose_dict = {'version': 1.3, 'people': []}
            with open(out_path, 'w') as f:
                json.dump(openpose_dict, f)
            continue

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks is None:
            openpose_dict = {'version': 1.3, 'people': []}
        else:
            coco25 = _pose33_to_coco25(result.pose_landmarks.landmark, w, h)
            openpose_dict = _to_openpose_json_dict(coco25)
            detected += 1

        with open(out_path, 'w') as f:
            json.dump(openpose_dict, f)

    pose.close()

    return {
        'total_images': total,
        'detected_images': detected,
        'skipped_existing': skipped,
        'output_folder': output_folder,
    }
