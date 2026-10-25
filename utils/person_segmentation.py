import cv2
import numpy as np
import onnxruntime as ort


class PersonSegmenter:
    def __init__(self, threshold: float = 0.5):
        """Initialize the PersonSegmenter object properties.

        Args:
            threshold (float, optional): Minimum person probability for a pixel to be in the mask. Defaults to 0.5.
        """
        self.input_size = 256
        self.person_class = 15  # index of "person" among the 21 Pascal VOC classes
        self.threshold = threshold

        self.model = self.__get_model()
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def __get_model(self, path="models/person_lraspp.onnx"):
        print(f"[INFO] Loading model from {path} ...")
        # use CUDA only if this ONNX Runtime installation actually provides it
        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        ort_session = ort.InferenceSession(path, providers=providers)
        print(f"[DONE] Model loaded successfully on {ort_session.get_providers()}!")
        return ort_session

    def predict(self, x: np.ndarray):
        output = self.model.run([self.output_name], {self.input_name: x})
        return output

    def process_image(self, x: np.ndarray):
        # resize to the model input size
        x = cv2.resize(
            x, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )

        # convert to float32 in [0, 1]
        x = x.astype(np.float32)
        x = x / 255.0

        # set channel first and add batch dimension
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Segment the people in an RGB image.

        Args:
            x (np.ndarray): RGB image.

        Returns:
            np.ndarray: Mask of the same size as the image, 255 for person pixels and 0 elsewhere.
        """
        height, width = x.shape[:2]
        logits = self.predict(self.process_image(x))[0][0]  # (21, H, W)

        # softmax across classes, then keep the person probability
        logits = logits - np.max(logits, axis=0, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=0, keepdims=True)
        person_probability = probabilities[self.person_class]
        mask = (person_probability >= self.threshold).astype(np.uint8) * 255

        # restore the original image resolution
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask


if __name__ == "__main__":
    model = PersonSegmenter()
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    mask = model(img)

    print(mask.shape, mask.dtype)
