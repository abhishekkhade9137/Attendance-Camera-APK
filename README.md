# Face Recognition Attendance System

A desktop application built with **Python** that uses your webcam and face recognition to automatically mark attendance. Registered faces are compared in real time using OpenCV's LBPH algorithm and all records are saved with timestamps to a CSV file.

---

## Screenshots

| Dashboard | Register Face | Mark Attendance |
|-----------|--------------|-----------------|
| View registered people, today's log, delete users | Live camera, take photo, confirm save | Real-time recognition with session log |

---

## Features

- **Dashboard**
  - Count of registered faces and today's attendance
  - Scrollable list of all registered people with individual **Delete** buttons
  - Scrollable table of today's attendance records (Name · Time · Date)
  - Refresh and Clear Today buttons

- **Register New Face**
  - Live webcam feed with face-detection overlay
  - Two-step flow: **Take Photo → Confirm & Save**
  - Retake option before saving
  - Automatically retrains the recognition model on save

- **Mark Attendance**
  - Real-time face recognition via webcam
  - Green bounding box for recognized faces, red for unknown
  - Each recognition is timestamped and saved to `attendance.csv`
  - Live **Session Log** panel showing every mark in the current session
  - Duplicate prevention — a person is only marked once per day

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `customtkinter` | Modern dark-mode UI framework |
| `opencv-python` | Camera capture, Haar Cascade face detection |
| `opencv-contrib-python` | LBPH Face Recognizer |
| `Pillow` | Image conversion for UI display |
| `numpy` | Numerical processing for model training |

---

## Project Structure

```
Attendance-Camera-APK/
|
|-- main.py              # Application entry point & all UI logic
|-- README.md            # Project documentation
|-- requirements.txt     # Python dependencies
|-- .gitignore
|
|-- models/
|   └── haarcascade_frontalface_default.xml   # Pre-trained face detection model (OpenCV)
|
└── data/
    |-- faces/           # Registered face images (.jpg) — created automatically
    |   |-- Abhishek.jpg
    |   └── ...
    └── attendance.csv   # Attendance log — created automatically
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Attendance-Camera-APK.git
cd Attendance-Camera-APK
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python main.py
```

> **Note:** Make sure your webcam is not blocked or in use by another application (e.g., video call software).

---

## How to Use

### Register a New Person
1. Click **Register Face** in the sidebar.
2. Position your face clearly in the camera frame.
3. Enter the person's name in the text field.
4. Click **Take Photo** to freeze the frame.
5. Click **Confirm & Save** — the face is saved and the model retrains automatically.

### Mark Attendance
1. Click **Mark Attendance** in the sidebar.
2. The camera will start and detect faces in real time.
3. A **green box** with the person's name appears on recognition.
4. Attendance is logged to `attendance.csv` with the current date and time.
5. Each person is only marked **once per day**.

### Manage Registered People
1. Go to the **Dashboard**.
2. The left panel lists all registered people.
3. Click the **Delete** button next to a name to remove them and retrain the model.

### View / Clear Attendance
1. Go to the **Dashboard**.
2. The right panel shows today's attendance log.
3. Click **Refresh** to reload data from disk.
4. Click **Clear Today** to wipe today's records.

---

## Attendance CSV Format

Records are stored in `data/attendance.csv` with the following columns:

```
Name, Time, Date
Abhishek, 09:15:32, 2026-07-08
Anshul, 09:17:44, 2026-07-08
```

---

## How Face Recognition Works

1. **Detection** — `haarcascade_frontalface_default.xml` (a pre-trained Haar Cascade classifier) scans each camera frame to locate face bounding boxes.
2. **Training** — When a face is registered, the image is saved to `data/faces/`. The **LBPH (Local Binary Patterns Histograms)** model is then trained on all saved images.
3. **Recognition** — During attendance, the live face ROI is compared against the trained model. If the confidence score is below 75, the face is considered a match and attendance is marked.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Black camera feed | Close other apps using the webcam (Zoom, Teams, etc.) |
| "Model trained successfully" repeats | Normal — model retrains each time you register or delete a face |
| Face not recognized | Re-register with better lighting and a clear, front-facing photo |
| Camera error message on Register screen | Restart the app and ensure webcam drivers are working |

---

## License

This project is open source and available under the [MIT License](LICENSE).
