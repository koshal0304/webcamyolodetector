# YOLO Vision Pro 🚀

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-000000?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

Real-time object detection web application using YOLOv8 models with Streamlit. Detect objects in images or through your webcam with state-of-the-art accuracy and speed!
<img width="1432" alt="Screenshot 2025-04-02 at 4 40 01 PM" src="https://github.com/user-attachments/assets/e63d9292-3269-4afe-9bc3-c50cd9c96f91" />



## Features ✨

- 🖼️ Image object detection with multiple YOLOv8 models
- 📸 Real-time webcam detection with OpenCV or WebRTC
- ⚡ Multiple model sizes (Nano, Small, Medium, Large)
- 🎚️ Adjustable confidence and IOU thresholds
- 📊 Detailed detection statistics and visualizations
- 💾 Export results as images or video recordings
- 🎨 Modern UI with custom styling and dark/light mode support

## Installation 🛠️

1. **Clone the repository**
```bash
git clone https://github.com/koshal0304/webcamyolodetector.git
cd webcamyolodetector
```
2. **Install dependencies**
 ```bash
pip install -r requirements.txt
```
3. **Download YOLOv8 ONNX models**
- Download models from Ultralytics YOLOv8 Releases.
- Place model files (yolov8n.onnx, yolov8s.onnx, etc.) in the project root.

## Usage 🚀
1. **Start the application**
```bash
streamlit run app.py
```
2. **Configure detection settings** in the sidebar:

- Select model size (Nano, Small, Medium, Large)
- Adjust confidence and IOU thresholds
- Choose between image upload or webcam mode

3. **Image Mode** 📷

- Upload any JPG/PNG image
- View detection results with bounding boxes
- Download processed image with detections

4. **Webcam Mode** 🌐

- Choose between OpenCV or WebRTC implementation
- Real-time object detection with FPS counter
- Record and download detection videos (optional)

## Model Zoo 🐘

| Model        | Size   | Speed (FPS) | Accuracy (mAP) | Use Case          |
|--------------|--------|-------------|----------------|-------------------|
| YOLOv8 Nano  | 4.3MB  | 120+        | 37.3           | Mobile & Edge     |
| YOLOv8 Small | 11.4MB | 80          | 44.9           | Balanced          |
| YOLOv8 Medium| 25.9MB | 50          | 50.2           | General Purpose   |
| YOLOv8 Large | 43.7MB | 30          | 52.9           | High Accuracy     |

**Copy-Paste Version:**
## Customization 🎨
1. **Add custom models**
- Convert custom YOLOv8 models to ONNX format

```bash
yolo export model=yolov8n.pt format=onnx
```
- Place in project root and update model paths in code

2. **Modify class labels**
- Edit `COCO_LABELS` list in app.py for custom datasets

3. **UI customization**
- Modify CSS styles in `load_custom_css()` function
- Add new visual elements using Streamlit components

