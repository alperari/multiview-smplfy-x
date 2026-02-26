import json
import cv2
import numpy as np

json_path = "dataset_example/image_data/rp_dennis_posed_004/keypoints/0300_keypoints.json"
# adjust extension/path if needed
img_path = "dataset_example/image_data/rp_dennis_posed_004/color/0300.jpg"
out_path = "kp_overlay_0300.jpg"

with open(json_path, "r") as f:
    data = json.load(f)

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

people = data.get("people", [])
if not people:
    print("No people found in JSON")
    cv2.imwrite(out_path, img)
    raise SystemExit

# Try common OpenPose keys in order
candidate_keys = [
    "pose_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
    "face_keypoints_2d",
]

colors = {
    "pose_keypoints_2d": (0, 255, 0),
    "hand_left_keypoints_2d": (255, 0, 0),
    "hand_right_keypoints_2d": (0, 0, 255),
    "face_keypoints_2d": (0, 255, 255),
}

for person in people:
    for k in candidate_keys:
        arr = person.get(k, [])
        if not arr:
            continue
        pts = np.array(arr, dtype=np.float32).reshape(-1,
                                                      3)  # x, y, confidence
        for x, y, c in pts:
            if c > 0.05:  # confidence threshold
                cv2.circle(img, (int(x), int(y)), 2, colors[k], -1)

cv2.imwrite(out_path, img)
print(f"Saved overlay to {out_path}")
