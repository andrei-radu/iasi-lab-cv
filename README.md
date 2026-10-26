# Computer Vision Applications on Jetson Nano

This project represents the second **IASI** laboratory (ro: _Inteligență Artificială pentru Sisteme Integrate_, en: _Artificial Intelligence for Integrated Systems_), for the **BIOSINF** (ro: _Tehnologii Multimedia în Aplicații de Biometrie și Securitatea Informației_ , en: _Multimedia Technologies in Biometrics and Information Security Applications_) master program.

The project contains three applications that run on the live camera feed:

| Application | Script | What it does |
| --- | --- | --- |
| [Rock Paper Scissors](#rock-paper-scissors-game) | `main_rps.py` | Recognizes your hand gesture and plays rock-paper-scissors against the computer. |
| [Object Tracking](#object-tracking) | `main_track.py` | Follows an object you select with the mouse and draws the path it moved along. |
| [People Segmentation](#people-segmentation) | `main_segm.py` | Highlights the people in the image, or replaces the background behind them. |

All of them run on the Jetson Nano by default, and also on a regular computer with a webcam (see [Running on a desktop](#running-on-a-desktop)).



# Prerequisites

## Camera setup

First, check if the camera is working by running the following command:
```bash
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink
```

If the camera is working, you should see a window with the camera feed. Press `Ctrl+C` to close the window.

If the camera is not working, make sure the pins are configured correctly. Use the following command to check the pins:
```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

## Install and update pip package manager
Run the following commands to install and update the pip package manager:
```bash
sudo apt-get update
sudo apt-get install python3-pip
pip3 install --upgrade pip
```


## Install ONNX Runtime
The models are in the ONNX format, so we need to install the ONNX Runtime. Run the following commands to install the ONNX Runtime:
```bash
wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
```

## Install OpenCV and NumPy
Run the following commands to install OpenCV and NumPy:
```bash
pip3 install opencv-python==4.5.5.62 opencv-contrib-python==4.5.5.62 numpy==1.19.4
```
The `opencv-contrib-python` package provides the KCF and CSRT trackers used by the [Object Tracking](#object-tracking) application.

## Clone the repository
Jetson Nano should already have the git-cli installed. If not, run the following command:
```bash
sudo apt-get install git
```

Then clone the repository:
```bash
git clone https://github.com/andrei-radu/iasi-lab-cv.git
```

All the commands below are run from the root directory of the repository.

## Common parameters and controls
Every application accepts this parameter:

| Parameter | Default | Description |
| --- | --- | --- |
| `-p`, `--platform` | `jetson` | Where the camera image comes from. `jetson` reads the Jetson Nano camera connected to the CAM0 port, at 640x480 and 30 FPS. `desktop` reads the computer's default webcam, at its default resolution. |

The camera image is flipped horizontally, so it behaves like a mirror. Each application shows the result in a window called `video-feed`; press **Q** in that window to quit.


# Rock Paper Scissors Game

A MobileNetV3 model, trained on the [HaGRID](https://gitlab.ai.cloud.ru/rndcv/hagrid) hand gesture dataset, recognizes your gesture about three times per second. Each time it recognizes a new gesture that is a valid move, a round is played against a computer that picks its move at random. The window shows your move, the computer's move and the winner of the last round.

| Move | Gestures that count as this move |
| --- | --- |
| Rock | `fist` |
| Paper | `palm`, `stop` |
| Scissors | `peace`, `two up`, and their upside-down versions |

Any other gesture is ignored and the last result stays on screen. A round is played only when the gesture changes, so lower your hand (or make any other gesture) between rounds.

## Download the model
Run the following command in the root directory of the project to download the model:
```bash
wget -O models/MobileNetV3FF_small.onnx "https://github.com/andrei-radu/iasi-lab-cv/releases/download/model/MobileNetV3FF_small.onnx"
```

## How to run
```bash
python3 main_rps.py
```

| Parameter | Default | Description |
| --- | --- | --- |
| `-p`, `--platform` | `jetson` | See [Common parameters](#common-parameters-and-controls). |
| `-f`, `--framework` | `onnx` | Which version of the model runs. `onnx` runs `models/MobileNetV3FF_small.onnx` with ONNX Runtime. `torch` runs the PyTorch version, `models/MobileNetV3FF_small.pth`, which also needs `torch` and `torchvision` installed. |

| Key | Action |
| --- | --- |
| **Q** | Quit |


# Object Tracking

Select any object in the image with the mouse, and an OpenCV tracker follows it from frame to frame. The application draws a green box around the object and a line through the positions where its center has been, so you can see the path it moved along. This application does not need a model.

## How to run
```bash
python3 main_track.py
```

| Parameter | Default | Description |
| --- | --- | --- |
| `-p`, `--platform` | `jetson` | See [Common parameters](#common-parameters-and-controls). |
| `-t`, `--tracker` | `KCF` | Which tracking algorithm to use. `KCF` is fast, but loses the object more easily when it moves quickly, changes size or is partly hidden. `CSRT` follows the object more reliably, but is slower, so the video runs at a lower frame rate, especially on the Jetson Nano. |
| `-l`, `--max_length` | `300` | How many past positions the motion trail keeps. One position is added per frame, so `300` is about 10 seconds of movement at 30 FPS. |

| Key | Action |
| --- | --- |
| **R** | Select a new object. The video pauses; drag a box around the object, then press **ENTER** or **SPACE** to start tracking it, or **C** to cancel the selection. |
| **C** | Clear the motion trail. |
| **Q** | Quit. |

If the tracker loses the object, tracking stops. Press **R** to select it again.


# People Segmentation

An [LR-ASPP MobileNetV3](https://pytorch.org/vision/stable/models/lraspp.html) model labels every pixel of the image with one of 21 classes (the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) classes). The pixels labeled as `person` are tinted green, or, if you give it an image, everything else is replaced with that image, like a virtual background in a video call.

## Get the model
Run the following command in the root directory of the project to download the model:
```bash
wget -O models/person_lraspp.onnx "https://github.com/andrei-radu/iasi-lab-cv/releases/download/model/person_lraspp.onnx"
```


## How to run
```bash
python3 main_segm.py
```
To replace the background with one of the images in `images/`:
```bash
python3 main_segm.py -i images/moon.jpg
```

| Parameter | Default | Description |
| --- | --- | --- |
| `-p`, `--platform` | `jetson` | See [Common parameters](#common-parameters-and-controls). |
| `-t`, `--threshold` | `0.5` | How sure the model must be, from `0` to `1`, before a pixel counts as a person. Lower values cover more of the person, but also mark parts of the background by mistake. Higher values keep only the pixels the model is most sure about, so edges and limbs may be cut off. |
| `-i`, `--img` | none | Path to an image to show behind the people, for example `images/moon.jpg`. The image is stretched to the size of the camera image. Without this parameter, the people are tinted green and the background stays as it is. |
| `--max-fps` | `5` | How many times per second the segmentation runs, from `1` to `30`. The video still updates on every frame, reusing the last result in between. Higher values make the mask follow movement more closely, but need more computing power. On the Jetson Nano, a value that is too high slows down the whole video. |

| Key | Action |
| --- | --- |
| **Q** | Quit |


# Acknowledgments
- [HaGRID](https://gitlab.ai.cloud.ru/rndcv/hagrid), which is under the CC-by-SA licence.
- [torchvision](https://github.com/pytorch/vision), for the pretrained LR-ASPP MobileNetV3 segmentation model.

# License
This project is licensed under the [MIT License](LICENSE).
