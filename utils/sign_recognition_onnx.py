import cv2
import numpy as np
import onnxruntime as ort


class HandSignRecognizer:
    def __init__(self):
        """ Initialize the HandSignRecognizer object properties. """
        self.targets = {
            1: "call",
            2: "dislike",
            3: "fist",
            4: "four",
            5: "like",
            6: "mute",
            7: "ok",
            8: "one",
            9: "palm",
            10: "peace",
            11: "rock",
            12: "stop",
            13: "stop inverted",
            14: "three",
            15: "two up",
            16: "two up inverted",
            17: "three2",
            18: "peace inverted",
            19: "no gesture",
        }

        self.model = self.__get_model()
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        

    def __get_model(self, path='models/MobileNetV3FF_small.onnx', num_classes=18):
        print(f"[INFO] Loading model from {path} ...")
        provider = ['CUDAExecutionProvider']
        ort_session = ort.InferenceSession(path, providers=provider)
        print("[DONE] Model loaded successfully!")
        return ort_session
    
    
    def predict(self, x: np.ndarray):
        output = self.model.run([self.output_name], {self.input_name: x})
        return output
    
    
    def process_image(self, x: np.ndarray):
        # convert to float32
        x = x.astype(np.float32)
        x = x / 255.0
        
        # center crop and resize to 224px
        h, w, c = x.shape
        crop_size = min(h, w)
        start_x = h // 2 - crop_size // 2
        start_y = w // 2 - crop_size // 2
        x = x[start_x:start_x + crop_size, start_y:start_y + crop_size, :]
        x = cv2.resize(x, (224, 224))
        
        # set channel first and add batch dimension
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        return x
    
    
    def __call__(self, x: np.ndarray):
        x = self.process_image(x)
        logits = self.predict(x)[0]
        class_ = np.argmax(logits)
        return self.targets[class_ + 1]
    
    
    
if __name__ == '__main__':
    model = HandSignRecognizer()
    img = np.random.rand(640, 480, 3)
    output = model(img)
    
    print(output)
    