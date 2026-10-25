import cv2
import numpy as np
from typing import Union


class Display:
    def __init__(self) -> None:
        """
        Initialize the display object properties.
        """
        self.window_name = "video-feed"
        self.pos = (50, 50)
        self.text_options = {
            "fontFace": cv2.FONT_HERSHEY_PLAIN,
            "fontScale": 2,
            "color": (0, 255, 0),
            "thickness": 2,
            "lineType": 2,
        }

    def show(self, image: np.ndarray, text: Union[str, None] = None) -> int:
        """Display the image with optional text overlay.

        Args:
            image (np.ndarray): Image to display.
            text (str or None, optional): Text to overlay on the image. Defaults to None.

        Returns:
            int: Code of the key pressed while displaying, or 255 if none.
        """
        if text is not None:
            text = text.split("\n")
            for idx, t in enumerate(text):
                text_pos = (self.pos[0], (idx + 1) * self.pos[1])
                image = cv2.putText(image, t, text_pos, **self.text_options)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, image)
        return cv2.waitKey(1) & 0xFF


def draw_trajectory(frame, trajectory):
    """Draw lines connecting the tracked object's previous centers."""
    for i in range(1, len(trajectory)):
        # Older positions are drawn more faintly/thinner
        thickness = max(1, int(3 * i / len(trajectory)))

        cv2.line(
            frame,
            trajectory[i - 1],
            trajectory[i],
            (0, 0, 255),  # Red trail in BGR
            thickness,
            cv2.LINE_AA,
        )


def overlay_mask(frame, mask, color=(0, 255, 0)):
    """Tint the pixels inside the mask with a color."""
    result = frame.copy()
    selected = mask > 0
    result[selected] = (0.5 * result[selected] + 0.5 * np.array(color)).astype(
        np.uint8
    )
    return result


def replace_background(frame, mask, background):
    """Replace the pixels outside the mask with a background image."""
    background = cv2.resize(
        background, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    result = frame.copy()
    selected = mask == 0
    result[selected] = background[selected]
    return result


if __name__ == "__main__":
    from camera import Camera

    cam = Camera()
    cam.start()

    display = Display()

    while True:
        img = cam.get_image()
        display.show(img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
