from typing import Dict

import torch
import torchvision
from torch import Tensor, nn


class MobileNetV3(nn.Module):
    """
    Torchvision two headed MobileNet V3 configuration
    """

    def __init__(
        self, num_classes: int, size: str = "small", pretrained: bool = False, freezed: bool = False, ff: bool = False
    ) -> None:
        """
        Torchvision two headed MobileNet V3 configuration

        Parameters
        ----------
        num_classes : int
            Number of classes for each task
        size : str
            Size of MobileNetV3 ('small' or 'large')
        pretrained : bool
            Using pretrained weights or not
        freezed : bool
            Freezing model parameters or not
        ff : bool
            Enable full frame mode
        """

        super(MobileNetV3, self).__init__()
        self.ff = ff

        if size == "small":
            torchvision_model = torchvision.models.mobilenet_v3_small(pretrained)
            in_features = 576
            out_features = 1024
        else:
            torchvision_model = torchvision.models.mobilenet_v3_large(pretrained)
            in_features = 960
            out_features = 1280

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(torchvision_model.features, torchvision_model.avgpool)

        self.gesture_classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=out_features, out_features=num_classes),
        )
        if not self.ff:
            self.leading_hand_classifier = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=out_features, out_features=2),
            )

        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Dict:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        gesture = self.gesture_classifier(x)

        if self.ff:
            return {"gesture": gesture}
        else:
            leading_hand = self.leading_hand_classifier(x)
            return {"gesture": gesture, "leading_hand": leading_hand}
        


class HandSignRecognizer:
    def __init__(self):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __get_model(self, path='models/MobileNetV3FF_small.pth', num_classes=18, size='small'):
        pack = torch.load(path, map_location=torch.device('cpu'))
        model = MobileNetV3(num_classes=num_classes, size=size, ff=True)
        model.load_state_dict(pack['state_dict'])
        model.to(self.device)
        model = model.eval()
        return model
    
    def predict(self, img):
        h, w, _ = img.shape
        crop_size = min(h, w) // 224
        img = img / 255
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = torchvision.transforms.functional.center_crop(img, (crop_size*224, crop_size*224))
        img = torchvision.transforms.functional.resize(img, (224, 224))
        img = img.unsqueeze(0).float()
        with torch.no_grad():
            pred = self.model(img.to(self.device))
        gesture = torch.argmax(pred['gesture'], dim=1).item()
        gesture = self.targets[gesture+1]
        return gesture
        

if __name__ == '__main__':
    import time
    from camera import Camera
    from display import Display


    cam = Camera()
    cam.start()
    disp = Display()
    hand_recog = HandSignRecognizer()

    while True:
        img = cam.get_image()
        gesture = hand_recog.predict(img)
        disp.show(img, text=gesture)

