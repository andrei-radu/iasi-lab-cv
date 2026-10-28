import argparse

from utils.camera import Camera
from utils.display import Display
from utils.game_logic import RockPaperScissors


def get_model(framework):
    if framework == "onnx":
        from utils.sign_recognition_onnx import HandSignRecognizer
    elif framework == "torch":
        from utils.sign_recognition import HandSignRecognizer
    else:
        raise ValueError("Invalid framework. Must be onnx or torch")
    return HandSignRecognizer()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument(
    #     "-p",
    #     "--platform",
    #     type=str,
    #     default="jetson",
    #     help="Choose the platform: jetson or desktop",
    # )
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
        "-f",
        "--framework",
        type=str,
        default="onnx",
        help="Choose the framework: onnx or torch",
    )
    args = arg_parser.parse_args()

    camera = Camera(platform=args.platform, device_id=args.camera_id)
    camera.start()
    display = Display()
    sign_recog = get_model(args.framework)
    game = RockPaperScissors()

    last_state = "ok"  # placeholder for last recognized sign
    text = None

    frame_counter = 0
    while True:
        img = camera.get_image()

        if frame_counter % 10 == 0:
            sign = sign_recog(img)

            if sign != last_state:
                last_state = sign
                results = game.round(sign)

                # only a valid move updates the text, the frame is shown either way
                if "error" not in results:
                    text = f"Player: {results['player_choice']}\n"
                    text += f"Computer: {results['computer_choice']}\n"
                    text += f"Winner is {results['who_wins']}\n\n"

        if display.show(img, text) == ord("q"):
            break
        frame_counter += 1

    camera.stop()
