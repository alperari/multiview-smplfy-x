import argparse

from keypoint_extractor import extract_keypoints_from_folder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate OpenPose-style keypoint JSONs for all images in a folder"
    )
    parser.add_argument("--image_folder", required=True,
                        help="Folder containing input images")
    parser.add_argument("--output_folder", required=True,
                        help="Folder to save *_keypoints.json files")
    parser.add_argument("--backend", default="mediapipe", choices=["mediapipe"],
                        help="Keypoint backend")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing keypoint JSON files")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5,
                        help="Minimum detection confidence for backend")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                        help="Minimum tracking confidence for backend")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    stats = extract_keypoints_from_folder(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        backend=args.backend,
        overwrite=args.overwrite,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    print("Done.")
    print("Total images:", stats["total_images"])
    print("Detected images:", stats["detected_images"])
    print("Skipped existing:", stats["skipped_existing"])
    print("Output folder:", stats["output_folder"])
