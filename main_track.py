import argparse
import cv2


from utils.camera import Camera
from utils.display import Display, draw_trajectory
from utils.track_algorithms import create_tracker


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="jetson",
        choices=["jetson", "desktop"],
        help="Choose the platform: jetson or desktop",
    )
    arg_parser.add_argument(
        "-t",
        "--tracker",
        type=str,
        default="KCF",
        choices=["CSRT", "KCF"],
        help="Choose the tracker type: CSRT or KCF",
    )
    arg_parser.add_argument(
        "-l",
        "--max_length",
        type=int,
        default=300,
        help="Maximum length of the motion trail",
    )

    args = arg_parser.parse_args()

    camera = Camera(platform=args.platform)
    camera.start()
    display = Display()
    tracking = False
    tracker = create_tracker(tracker_type=args.tracker)
    trajectory = []

    while True:
        frame = camera.get_image()
        display_frame = frame.copy()
        if tracking and tracker is not None:
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = map(int, bbox)
                trajectory.append((x + w // 2, y + h // 2))
                if len(trajectory) > args.max_length:
                    trajectory.pop(0)
                draw_trajectory(display_frame, trajectory)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(
                    display_frame,
                    "Tracking lost - press R to retry",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                tracking = False
                tracker = None

        cv2.putText(
            display_frame,
            "R: Select object | C: Clear trail | Q: Quit",
            (10, display_frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        key = display.show(display_frame)
        if key == ord("q"):
            break

        elif key == ord("c"):
            trajectory.clear()
            print("Motion trail cleared.")

        elif key == ord("r"):
            tracking = False
            tracker = None
            trajectory.clear()

            # Select a new object with the mouse, in the same window as the feed
            roi = cv2.selectROI(
                display.window_name,
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                fromCenter=False,
                showCrosshair=True,
            )

            x, y, w, h = roi

            if w > 0 and h > 0:
                tracker = create_tracker(tracker_type=args.tracker)
                tracker.init(frame, roi)
                tracking = True
                print("Tracking started.")

    camera.stop()

    # ############################################

    # cap = cv2.VideoCapture(CAMERA_ID)

    # if not cap.isOpened():
    #     raise RuntimeError("Could not open camera.")

    # tracker = None
    # tracking = False
    # trajectory = []

    # window_name = "OpenCV Object Tracking"
    # cv2.namedWindow(window_name)

    # print("Controls:")
    # print("  R - Select an object / restart tracking")
    # print("  C - Clear the motion trail")
    # print("  Q - Quit")

    # while True:
    #     ok, frame = cap.read()
    #     # flip the frame horizontally for a mirror effect
    #     frame = cv2.flip(frame, 1)

    #     if not ok:
    #         print("Could not read a frame from the camera.")
    #         break

    #     display = frame.copy()

    #     if tracking and tracker is not None:
    #         success, bbox = tracker.update(frame)

    #         if success:
    #             x, y, w, h = map(int, bbox)

    #             # Center of the tracked bounding box
    #             center = (x + w // 2, y + h // 2)

    #             # Save this frame's center position
    #             trajectory.append(center)

    #             # Keep only the most recent positions
    #             if len(trajectory) > MAX_TRAJECTORY_LENGTH:
    #                 trajectory.pop(0)

    #             # Draw the motion trail
    #             draw_trajectory(display, trajectory)

    #             # Draw the bounding box
    #             cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #         else:
    #             cv2.putText(
    #                 display,
    #                 "Tracking lost - press R to retry",
    #                 (20, 35),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.7,
    #                 (0, 0, 255),
    #                 2,
    #             )

    #             tracking = False
    #             tracker = None

    #     # Show controls on screen
    #     cv2.putText(
    #         display,
    #         "R: Select object | C: Clear trail | Q: Quit",
    #         (10, display.shape[0] - 15),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         (255, 255, 255),
    #         1,
    #     )

    #     cv2.imshow(window_name, display)

    #     key = cv2.waitKey(1) & 0xFF

    #     if key == ord("q"):
    #         break

    #     elif key == ord("c"):
    #         trajectory.clear()
    #         print("Motion trail cleared.")

    #     elif key == ord("r"):
    #         # Stop current tracking and reset the trail
    #         tracking = False
    #         tracker = None
    #         trajectory.clear()

    #         # Select a new object with the mouse
    #         roi = cv2.selectROI(
    #             window_name, frame, fromCenter=False, showCrosshair=True
    #         )

    #         x, y, w, h = roi

    #         if w > 0 and h > 0:
    #             tracker = create_tracker()
    #             tracker.init(frame, roi)
    #             tracking = True
    #             print("Tracking started.")

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
