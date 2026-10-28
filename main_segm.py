import argparse

import cv2

from utils.camera import Camera
from utils.display import Display, overlay_mask, replace_background
from utils.person_segmentation import PersonSegmenter

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="usb",
        help="Choose the platform: usb or csi",
    )
    arg_parser.add_argument(
        "-c",
        "--camera-id",
        type=int,
        default=0,
        help="Camera ID for camera (default: 0)",
    )
    arg_parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum person probability for a pixel to be segmented",
    )
    arg_parser.add_argument(
        "-i",
        "--img",
        type=str,
        default=None,
        help="Path to an image that replaces the background",
    )
    arg_parser.add_argument(
        "--max-fps",
        type=int,
        default=5,
        help="Maximum number of times per second the segmentation runs",
    )
    args = arg_parser.parse_args()

    camera = Camera(platform=args.platform, device_id=args.camera_id)
    camera.start()
    display = Display()
    segmenter = PersonSegmenter(threshold=args.threshold)

    background = None
    if args.img is not None:
        background = cv2.imread(args.img)
        if background is None:
            raise FileNotFoundError(f"Could not read image {args.img}")
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    frame_counter = 0
    while True:
        img = camera.get_image()

        if frame_counter % (camera.fps // args.max_fps) == 0:
            mask = segmenter(img)

        if background is not None:
            img = replace_background(img, mask, background)
        else:
            img = overlay_mask(img, mask)

        if display.show(img) == ord("q"):
            break
        frame_counter += 1

    camera.stop()
