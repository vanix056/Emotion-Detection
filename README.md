# Emotion Detection

## Overview

Real-time facial emotion detection system that captures live webcam feed, identifies faces using OpenCV's Haar Cascade classifier, and predicts the dominant emotion using the DeepFace deep learning library. The system overlays detected bounding boxes and the predicted emotion label directly onto the video stream, making it suitable for rapid prototyping in human-computer interaction, sentiment analysis, and behavioral monitoring applications. Processing is optimized by running emotion inference every 5 frames, balancing accuracy with performance on standard hardware.

## Key Features

- **Live webcam capture** — supports primary and fallback camera devices automatically
- **Haar Cascade face detection** — lightweight, real-time face localization using OpenCV
- **Deep learning emotion analysis** — leverages DeepFace to classify 7 emotions: angry, disgust, fear, happy, sad, surprise, and neutral
- **Performance-aware inference** — emotion analysis runs every 5 frames to reduce CPU/GPU load
- **On-screen annotation** — bounding boxes and dominant emotion label rendered directly on the video feed
- **Graceful exit** — press `q` to cleanly release the camera and close all windows

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Computer Vision | OpenCV (`cv2`) |
| Emotion Analysis | DeepFace |
| Deep Learning Backend | TensorFlow 2.x |
| Notebook Environment | Jupyter Notebook |
| Face Detection Model | Haar Cascade (`haarcascade_frontalface_default.xml`) |

## Installation

### Prerequisites

- Python 3.9–3.11
- A connected webcam

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/MAbdullahWaqar/Emotion-Detection.git
cd Emotion-Detection

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install opencv-python deepface tensorflow

# 4. If using TensorFlow >= 2.16, also install tf-keras
pip install tf-keras
```

## Usage

### Run via Jupyter Notebook

```bash
jupyter notebook Emotion_Detector.ipynb
```

Open the notebook in your browser and execute the single code cell. The webcam window will launch automatically.

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit and close the webcam window |

### Expected Output

A live video window titled **"Emotion Detection"** will appear, showing:
- A green rectangle drawn around each detected face
- The predicted dominant emotion (e.g., `happy`, `neutral`, `sad`) displayed in red at the top-left of the frame

## Model Architecture

DeepFace is used as a high-level abstraction over several pre-trained deep learning models. For emotion analysis, it internally uses a lightweight CNN-based emotion classifier trained on the FER-2013 dataset. The face detection pipeline uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) for fast, CPU-friendly bounding box prediction prior to passing crops to DeepFace.

| Component | Model / Method |
|---|---|
| Face Detection | Haar Cascade (OpenCV) |
| Emotion Classification | DeepFace Emotion CNN |
| Inference Mode | `enforce_detection=False` (robust to partial faces) |

## Dataset

This project uses DeepFace's pre-trained emotion model, which was originally trained on the **FER-2013** dataset:

- **Source**: Kaggle / originally from ICML 2013 Challenges in Representation Learning
- **Classes**: angry, disgust, fear, happy, sad, surprise, neutral
- **Format**: 48×48 pixel grayscale facial images

No additional dataset download is required for inference.

## Project Structure

```
Emotion-Detection/
├── Emotion_Detector.ipynb            # Main notebook — webcam emotion detection pipeline
├── haarcascade_frontalface_default.xml  # Haar Cascade model for face detection
├── happy.jpg                         # Sample image
├── images.jpeg                       # Sample image
└── README.md
```

## Configuration

The following parameters can be adjusted directly in the notebook cell:

| Parameter | Default | Description |
|---|---|---|
| `cv2.VideoCapture(1)` | `1` (fallback: `0`) | Camera device index |
| `CAP_PROP_FRAME_WIDTH` | `640` | Capture frame width in pixels |
| `CAP_PROP_FRAME_HEIGHT` | `480` | Capture frame height in pixels |
| `frame_count % 5` | `5` | Run emotion inference every N frames |
| `enforce_detection` | `False` | Allow analysis even when face confidence is low |

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch from `main`.
2. Keep changes focused and well-documented.
3. Test your changes locally before submitting a pull request.
4. Open a pull request with a clear description of the change and its motivation.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed by **Muhammad Abdullah Waqar**. For questions or suggestions, open an issue on the [GitHub repository](https://github.com/MAbdullahWaqar/Emotion-Detection).
