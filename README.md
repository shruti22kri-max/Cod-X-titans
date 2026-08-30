# Underwater Sonar Object Detection Prototype

An AI-powered Side-Scan Sonar (SSS) prototype web application for ocean cleanup and underwater target detection.

---

## 🎯 Target Detection Capabilities

The system detects both critical environmental hazards and large maritime structures:
1. **Small Object Detection**: Compact lost equipment, small metal traps, canisters, pinpoint acoustic targets.
2. **Unknown Anomaly Detection**: Unidentified seabed reverberation signatures, acoustic anomalies, seafloor disturbances.
3. **Ghost Net Detection**: Abandoned/discarded commercial fishing nets draped over seabed contours.
4. **Major & Huge Maritime Objects**:
   - **Sunken Shipwrecks / Vessels**: Large hull structures, barges, submerged boats.
   - **Heavy Steel Rods & Structures**: Construction debris, structural steel, framing.
   - **Submerged Pipelines**: Underwater pipelines and conduits.
5. **Commercial Traps & Marine Debris**: Crab pots, wire cages, discarded tires, and general marine debris.

---

## 🧪 Included Demo Test Samples

The prototype comes with 3 synthetic sonar test scans:
- **Sample 1 (`sample_small_object.jpg`)**: Small Object Detection (pinpoint acoustic echo with tight acoustic shadow).
- **Sample 2 (`sample_unknown_anomaly.jpg`)**: Unknown Anomaly Detection (irregular perturbation contour and acoustic halo).
- **Sample 3 (`sample_ghost_net.jpg`)**: Ghost Net Detection (criss-crossing filament mesh drape with floaters and tangled shadows).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### 3. Run Automated Tests
```bash
python test_api.py
```

---

## 📡 API Specification

### Detection Endpoint
`POST /predict`

- **Request**: `multipart/form-data` with form field `file` (JPG, JPEG, or PNG image).
- **Response**: JSON matching the standard schema:

```json
{
  "detections": [
    {
      "class": "small_object",
      "confidence": 0.94,
      "x": 378,
      "y": 216,
      "width": 58,
      "height": 48
    },
    {
      "class": "sunken_shipwreck",
      "confidence": 0.98,
      "x": 216,
      "y": 168,
      "width": 432,
      "height": 228
    }
  ]
}
```

---

## 🧠 Connecting a Real AI Model

The model integration is completely isolated in **`backend/model_service.py`**.

When you have a trained PyTorch, Ultralytics YOLO, or ONNX model:

1. Open `backend/model_service.py`.
2. Set:
   ```python
   REAL_MODEL_ENABLED = True
   MODEL_PATH = "weights/best_sonar_model.pt"
   ```
3. Implement `load_real_model()` and `run_real_model_inference()` to run inference on the image and return the bounding box list.

---

## 📁 Project Structure

```
marineDebries/
├── backend/
│   ├── app.py              # FastAPI server handling /predict, /api/samples & static frontend serving
│   ├── model_service.py    # Sonar detection engine (Mock/Demo & Real Model hooks)
│   └── sample_generator.py # Generates synthetic side-scan sonar demo images
├── frontend/
│   ├── index.html          # SeaGuard Single-Page Application
│   ├── styles.css          # Marine oceanography dark theme & multi-class HUD bounding box styling
│   └── app.js              # Upload handling, API caller, and dynamic canvas bounding box renderer
├── sample_images/          # Demo side-scan sonar images (Small Objects, Anomalies, Ghost Nets)
├── run.py                  # One-click startup script with auto-browser opening
├── requirements.txt        # Backend dependencies
├── test_api.py             # 6-suite automated test verification script
└── README.md               # Documentation
```
