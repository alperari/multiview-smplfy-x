import argparse

from camera_estimator import estimate_cameras_from_folder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate camera parameters for all images in a folder using COLMAP"
    )
    parser.add_argument("--image_folder", required=True,
                        help="Folder containing input images")
    parser.add_argument("--output_path", required=True,
                        help="Output camera parameter JSON path")
    parser.add_argument("--backend", default="colmap", choices=["colmap"],
                        help="Camera estimation backend")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file")
    parser.add_argument("--colmap_binary", default="colmap",
                        help="COLMAP executable path or command name")
    parser.add_argument("--colmap_work_dir", default="",
                        help="COLMAP temporary work directory")
    parser.add_argument("--colmap_matcher", default="exhaustive",
                        choices=["exhaustive", "sequential"],
                        help="COLMAP matcher type")
    parser.add_argument("--colmap_camera_model", default="SIMPLE_PINHOLE",
                        help="COLMAP camera model")
    parser.add_argument("--colmap_single_camera", default=True,
                        type=lambda x: x.lower() in ["true", "1"],
                        help="Use shared intrinsics across all images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    stats = estimate_cameras_from_folder(
        image_folder=args.image_folder,
        output_path=args.output_path,
        backend=args.backend,
        overwrite=args.overwrite,
        colmap_binary=args.colmap_binary,
        colmap_work_dir=args.colmap_work_dir,
        colmap_matcher=args.colmap_matcher,
        colmap_camera_model=args.colmap_camera_model,
        colmap_single_camera=args.colmap_single_camera,
    )

    print("Done.")
    print("Num cameras:", stats["num_cameras"])
    print("Num registered images:", stats.get("num_registered_images", "n/a"))
    print("Used cache:", stats["used_cache"])
    print("Output path:", stats["output_path"])
