# People Counting System

_A computer vision project that leverages AI/ML techniques for real-time people counting using video streams._

---

## Description

This project implements a robust **people counting mechanism** using deep learning-based object detection (YOLOv3). It processes video input and counts individuals in real time—ideal for smart surveillance, retail footfall analytics, and public safety monitoring.

---

## Features

- **Deep Learning Detection**  
  Uses YOLOv3 to accurately detect and count people in images or video streams.
- **Real-Time Processing**  
  Capable of live inference with webcam or video files.
- **Configurable Model**  
  Easily switch detection classes using `coco.names` and update YOLO cfg for different environments.
- **Scalable Architecture**  
  Ready for deployment as edge or cloud service for larger camera networks.
- **Open Source and Extensible**  
  Modify or extend the Python code (`people_counter.py`) for analytics, alerts, or dashboard integration.

---

## Technologies Used

- **Python 3.8+**
- **OpenCV** – Real-time computer vision
- **YOLOv3** – Deep learning object detection
- **NumPy** – Efficient matrix computation

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the Repository**
    ```
    git clone https://github.com/YOUR_USERNAME/people-counting-system.git
    cd people-counting-system
    ```

2. **Install Dependencies**
    ```
    pip install opencv-python numpy
    ```

3. **Download YOLOv3 Weights**
    - Download `yolov3.weights` from [YOLO website](https://pjreddie.com/darknet/yolo/)
    - Place it in your project directory.

4. **Run the People Counter**
    ```
    python people_counter.py --source path/to/video.mp4
    ```
    (Change `--source` to a webcam index or video/image file as needed.)

---

## Project Structure
people-counting-system/
├── coco.names # COCO dataset class names
├── people_counter.py # Main Python script
├── yolov3.cfg # YOLOv3 configuration
├── yolov3.weights # YOLOv3 trained weights (add after download)
├── README.md # Project documentation
├── LICENSE # MIT License


---

## Demo
![879b6fa5-7679-44a2-8e49-df7f7dfea33c](https://github.com/user-attachments/assets/1ce33f99-ce47-4ef7-abb9-6a2ef60c80b1)
![4085558b-7e3d-4c49-8e0f-38c0c19c3578](https://github.com/user-attachments/assets/9fd88d8a-7d45-41af-8f62-8b9b50984165)
![dd3af16a-16f2-40e3-9d96-1f0d00209dce](https://github.com/user-attachments/assets/c81e96c4-2ef1-495a-a0e3-72b433951dce)

---

## License

MIT License

---

## Maintainer

SANYAM GARG
gargsanyam217@gmail.com

---



