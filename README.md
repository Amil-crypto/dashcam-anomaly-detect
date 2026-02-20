# dashcam-anomaly-detect
Real-time road anomaly detection using dashcam footage, optimized for Raspberry Pi with TensorFlow Lite. Detects potholes, cracks, and other road hazards efficiently on-device with minimal CPU usage.
Features

Detect potholes, cracks, and speed bumps in real-time.

Uses YOLOv3 for accurate and fast object detection.

Supports custom datasets for training.

Exportable to ONNX or TFLite for deployment on edge devices.

Easily integrates with OpenCV for live camera detection.
Installation

Clone the repository:

git clone https://github.com/<your-username>/road-anomaly-detection.git
cd road-anomaly-detection

Create and activate a Python virtual environment:

python3 -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

Install dependencies:

pip install -r requirements.txt

Dependencies include:

torch (PyTorch)

opencv-python

numpy

matplotlib

Dataset Preparation

Place training and validation images in dataset/images/.

Place YOLOv3 .txt annotation files in dataset/labels/.

Update dataset/classes.txt with class names (one per line):

pothole
crack
speed_bump

YOLOv3 .txt annotation format:

<class_id> <x_center> <y_center> <width> <height>

(All values normalized between 0 and 1.)

Training YOLOv3
python train.py --data dataset/ --cfg cfg/yolov3-custom.cfg --weights weights/yolov3.weights --epochs 50

Trains YOLOv3 on your custom dataset.

Outputs:

weights/yolov3_custom_final.weights → Trained weights for inference.

Real-Time Detection
python detect.py --weights weights/yolov3_custom_final.weights --source dashcam.mp4

Replace dashcam.mp4 with a video file or live camera input (0 for default camera).

The script draws bounding boxes and class labels on detected anomalies.

Deployment on Raspberry Pi 4B

Copy the trained YOLOv3 weights and detect.py to Raspberry Pi.

Install dependencies:

sudo apt update
sudo apt install python3-pip
pip3 install torch torchvision opencv-python numpy

Run real-time detection with a connected camera:

python3 detect.py --weights yolov3_custom_final.weights --source 0
License

This project is licensed under the MIT License.

Acknowledgements

YOLOv3 Darknet

OpenCV for video processing

PyTorch for training and inference
