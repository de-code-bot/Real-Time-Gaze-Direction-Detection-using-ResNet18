# 👁️ Real-Time Gaze Direction Detection

This project is a simple implementation of a **real-time gaze detection system** using deep learning.
It uses your webcam to track eye movement and predict where you're looking — like left, right, up, down, etc.

Under the hood, it uses a **ResNet18-based model** built with PyTorch to estimate gaze direction through pitch and yaw angles.

---

## 🚀 What this project does

* Detects face and eyes using OpenCV
* Predicts gaze direction in real-time
* Converts predictions into 8 directions (UP, DOWN, LEFT, RIGHT, etc.)
* Displays results with arrows, compass, and live stats
* Runs smoothly on a webcam

---

## 🧠 How it works (simple explanation)

Instead of directly guessing directions, the model predicts:

* **Pitch** → up/down movement
* **Yaw** → left/right movement

These values are then converted into directions like:

* UP
* DOWN
* LEFT
* RIGHT
* and diagonals

This approach makes the system more flexible and accurate.

---

## 🗂️ Project Structure

```bash
.
├── gaze_live_demo.py                         # Run this for real-time detection
├── gaze_estimation_training_resnet18.ipynb  # Training notebook
├── .gitignore
├── .gitattributes
```

---

## ⚙️ Requirements

Make sure you are using:

👉 **Python 3.11**

Install required libraries:

```bash
pip install torch torchvision opencv-python numpy pillow
```

---

## ▶️ How to run

Just run:

```bash
python gaze_live_demo.py
```

You can also pass options if needed:

```bash
python gaze_live_demo.py --model gaze_model.pth --camera 0
```

---

## 📦 Dataset

This project is based on the **MPIIGaze dataset**.

Since it’s quite large, it’s not included here.
You can download it from:

https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz

---

## 🔧 Tech Used

* Python
* PyTorch
* OpenCV
* NumPy

---

## 💡 Future ideas

* Improve accuracy with bigger models
* Add head pose detection
* Turn this into a web or mobile app

---

## 🙌 Final Note

This was built as a learning + practical project to explore **computer vision and deep learning in real-time applications**.

If you like it, feel free to ⭐ the repo!
