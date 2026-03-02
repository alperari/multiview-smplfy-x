import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np


CANDIDATE_KEYS = [
    "pose_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
    "face_keypoints_2d",
]

COLORS = {
    "pose_keypoints_2d": (0, 255, 0),
    "hand_left_keypoints_2d": (255, 0, 0),
    "hand_right_keypoints_2d": (0, 0, 255),
    "face_keypoints_2d": (0, 255, 255),
}

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image_path(image_folder, stem):
    for ext in IMG_EXTENSIONS:
        candidate = osp.join(image_folder, stem + ext)
        if osp.exists(candidate):
            return candidate
    return None


def draw_keypoints_on_image(img, data, conf_thresh):
    people = data.get("people", [])
    if not people:
        return img

    for person in people:
        for key in CANDIDATE_KEYS:
            arr = person.get(key, [])
            if not arr:
                continue
            pts = np.array(arr, dtype=np.float32).reshape(-1, 3)
            for x, y, conf in pts:
                if conf > conf_thresh:
                    cv2.circle(img, (int(x), int(y)), 2, COLORS[key], -1)
    return img


def visualize_folder(keypoint_folder, image_folder, output_folder, conf_thresh=0.05):
    os.makedirs(output_folder, exist_ok=True)

    keypoint_files = sorted(
        fn for fn in os.listdir(keypoint_folder)
        if fn.endswith("_keypoints.json")
    )

    saved = 0
    missing_image = 0

    for kp_file in keypoint_files:
        kp_path = osp.join(keypoint_folder, kp_file)
        stem = kp_file.replace("_keypoints.json", "")
        img_path = find_image_path(image_folder, stem)

        if img_path is None:
            print(f"[WARN] Image not found for {kp_file}")
            missing_image += 1
            continue

        with open(kp_path, "r") as f:
            data = json.load(f)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            missing_image += 1
            continue

        out_img = draw_keypoints_on_image(img, data, conf_thresh)
        out_path = osp.join(output_folder, f"{stem}_kp_overlay.jpg")
        cv2.imwrite(out_path, out_img)
        saved += 1

    print(f"Done. Saved overlays: {saved}")
    print(f"Skipped (missing/unreadable images): {missing_image}")
    print(f"Output folder: {output_folder}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize keypoints for a whole folder")
    parser.add_argument("--keypoint_folder", required=True,
                        help="Folder containing *_keypoints.json files")
    parser.add_argument("--image_folder", required=True,
                        help="Folder containing input images")
    parser.add_argument("--output_folder", required=True,
                        help="Folder to save overlay images")
    parser.add_argument("--conf_thresh", type=float, default=0.05,
                        help="Confidence threshold for drawing points")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_folder(
        keypoint_folder=args.keypoint_folder,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        conf_thresh=args.conf_thresh,
    )
