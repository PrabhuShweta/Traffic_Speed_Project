# Real-Time Traffic Monitoring & Speed Enforcement System

A production-grade computer vision pipeline built to detect Indian vehicle classes, track unique objects across frames using multi-object tracking, and calculate real-world vehicle speeds using perspective geometry transformation.

---

## 🚀 Features
* **Custom-Trained YOLOv8 Model:** Fine-tuned specifically for local road conditions to detect classes including cars, autos (`auto`), tempos (`tempo`), tractors (`tractor`), buses, and trucks.
* **Multi-Object Tracking:** Integrates ByteTrack to maintain consistent vehicle IDs across frame sequences.
* **Spatial Geometry & Speed Calculation:** Translates 2D pixel coordinates into a 3D real-world metric space using OpenCV perspective transformation matrices to calculate velocity ($v = \frac{\Delta d}{\Delta t}$ in km/h).
* **Automated Violation Flagging:** Real-time visual alerts for vehicles exceeding pre-configured speed thresholds.

---

## 🛠️ Tech Stack
* **Language:** Python 
* **Deep Learning:** Ultralytics YOLOv8, PyTorch
* **Computer Vision & Geometry:** OpenCV, Supervision
* **Configuration:** PyYAML

---

## 📂 Project Structure
```text
Traffic_Speed_Project/
│
├── config/settings.yaml       # Control center for paths, limits, and zones
├── utils/speed_calculator.py  # Perspective transform and speed-estimation logic
├── main.py                    # Real-time pipeline execution script
├── train_model.py             # Custom model fine-tuning script
└── requirements.txt           # Project dependencies
```

# ⚙️ Installation & Setup
1. Clone the repository:
git clone [https://github.com/YOUR_USERNAME/Traffic_Speed_Project.git](https://github.com/YOUR_USERNAME/Traffic_Speed_Project.git)
cd Traffic_Speed_Project

2. Create and activate a virtual environment:
python -m venv venv
# On Windows:
venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Run the tracking and speed enforcement system:
python main.py