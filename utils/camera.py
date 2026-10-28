import cv2
import numpy as np


class Camera:
    """Class for handling camera operations."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        platform: str = "usb",  # Platform on which the code is running
        device_id: int = 0,  # 0 or 1, corresponding to CAM port or /dev/videoX for USB camera
        flip: bool = False,  # Flip the image by 180 degrees
        width: int = 640,  # Width of the image, in pixels
        height: int = 480,  # Height of the image, in pixels
        fps: int = 30,  # Frames per second
    ):
        """Initialize the camera object properties.

        Args:
            platform (str, optional): Platform on which the code is running. Defaults to "usb". can be either "usb" or "csi".
            device_id (int, optional): ID of device, either 0 or 1, corresponding to CAM port. Defaults to 0.
            flip (bool, optional): Whether to flip the image by 180 degrees. Defaults to False.
            width (int, optional): Width of the image, in pixels. Defaults to 640.
            height (int, optional): Height of the image, in pixels. Defaults to 480.
            fps (int, optional): Frames per second. Defaults to 30.
        """
        assert platform in ["usb", "csi"], "Invalid platform. Must be usb or csi"
        self.platform = platform
        self.device_id = device_id
        self.flip = flip
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None  # placeholder for the camera object

    def start(self) -> None:
        """Open the camera and start capturing frames.

        Raises:
            RuntimeError: If the camera could not be opened.
        """

        if self.platform == "usb":
            pipeline = self.device_id
            api = cv2.CAP_ANY
        elif self.platform == "csi":  # unstable for many Jetpack versions
            pipeline = (
                f"nvarguscamerasrc sensor-id={self.device_id} ! "
                "video/x-raw(memory:NVMM), "
                f"width={self.width}, "
                f"height={self.height}, "
                "format=(string)NV12, "
                f"framerate={self.fps}/1 ! "
                "video/x-raw, format=(string)BGRx ! videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
            )
            api = cv2.CAP_GSTREAMER
        self.cap = cv2.VideoCapture(pipeline, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera on CAM{self.device_id}")
        _, _ = self.cap.read()  # activate camera

    def get_image(self) -> np.ndarray:
        """Get the image from the camera.

        Returns:
            np.ndarray: Image from the camera.
        """
        ok = False
        while not ok:
            ok, image = self.cap.read()  # read a frame from the camera
        if self.flip:
            image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def stop(self) -> None:
        """Stop capturing frames and release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    cam = Camera()
    cam.start()
    time.sleep(2)

    img = cam.get_image()

    plt.imshow(img)
    plt.axis("off")
    plt.show()

    cam.stop()

    print("Done!")
