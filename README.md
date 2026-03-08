🫘 NephroScan AI
Unified Kidney Analysis — 3-Model Deep Learning Pipeline

Stack: EfficientNet-B4 · FastAPI · Grad-CAM++ · ReportLab · Apple MPS

🔍 Overview

NephroScan AI is a unified AI pipeline that analyzes a single kidney CT scan using three deep learning models simultaneously to detect stones, classify kidney conditions, and assess cancer risk.

The system produces:

Multi-model predictions

Risk level assessment

Grad-CAM heatmap visualization

Auto-generated clinical PDF report

⚙️ AI Pipeline
CT Scan
   ↓
v1 — Stone Detector
Binary Classification
Accuracy: 99.2% | AUC: 1.0000

   ↓
v2 — 4-Class Kidney Classifier
Normal / Cyst / Stone / Tumour
Accuracy: 97.0%

   ↓
v3 — Cancer Detector
Cancer / Not Cancer
AUC: 0.9999 | Precision: 100%

   ↓
Risk Level + Grad-CAM Heatmap + Clinical PDF Report
📊 Model Performance
Model	Task	Accuracy	AUC	Key Metric
v1	Stone Detection	99.2%	1.0000	0 missed stones
v2	4-Class Classification	97.0%	0.9984	Tumour recall 92.7%
v3	Cancer Detection	99.6%	0.9999	Precision 100%
🧠 v2 Training Progress (4-Class Model)
Epoch	Accuracy	AUC	Tumour Recall	Notes
1	71.4%	0.9041	67.0%	Backbone frozen
3	76.0%	0.9278	63.4%	Backbone frozen
4	92.0%	0.9929	96.7%	Backbone unfrozen
5	95.6%	0.9960	88.8%	Fine-tuning
6	97.2%	0.9986	92.7%	⭐ Best checkpoint
🧠 v3 Training Progress (Cancer Detector)
Epoch	Accuracy	AUC	Cancer Recall	Precision	Notes
1	94.4%	0.9983	99.4%	77.9%	Backbone frozen
4	99.4%	0.9998	97.2%	99.4%	Backbone unfrozen
5	99.6%	0.9999	98.0%	100%	⭐ Best checkpoint
🚀 Quick Start
1️⃣ Activate Environment
cd '/Users/devaguru/Kidney Stone CNN/kidney-stone-cnn'
source .venv/bin/activate
2️⃣ Start the API Server
uvicorn api.unified_main:app --port 8000 --reload

Expected output:

Loading 3 models on mps...
v1 stone detector loaded ✅
v2 4-class classifier loaded ✅
v3 cancer detector loaded ✅
All 3 models ready ✅
3️⃣ Open the Dashboard
open nephroscan_unified.html
🔌 API Endpoints
Method	Endpoint	Description
POST	/predict	CT scan → predictions from all 3 models
POST	/report	Generate downloadable clinical PDF
GET	/health	Server health + device info
GET	/model-info	Model architecture and metrics
GET	/docs	Interactive Swagger UI
📡 Example API Response
{
  "v1": {
    "prediction": "no_stone",
    "has_stone": false,
    "confidence": 0.9821
  },
  "v2": {
    "prediction": "normal",
    "confidence": 0.9614,
    "clinical_note": "No abnormality detected. Routine follow-up recommended."
  },
  "v3": {
    "prediction": "not_cancer",
    "confidence": 0.9988
  },
  "risk_level": "NORMAL",
  "gradcam_heatmap": "base64 encoded image"
}
🏗 Model Architecture

All three models use the same backbone:

EfficientNet-B4 (ImageNet pretrained)

→ BatchNorm1d(1792)
→ Dropout(0.4)
→ Linear(1792 → 512)
→ GELU
→ BatchNorm1d(512)
→ Dropout(0.3)
→ Linear(512 → N)

Where:

N = 2 for v1 and v3

N = 4 for v2

🧪 Training Strategy

Training Phases

Phase 1 — Feature Extraction

Backbone frozen

Train classification head only

Learning rate: 1e-3

Phase 2 — Fine-Tuning

Backbone unfrozen

Backbone LR: 1e-4

Head LR: 1e-3

Other Training Details

Loss Function: Focal Loss (γ = 2.0)

Optimizer: AdamW

Sampling: WeightedRandomSampler

Calibration: Temperature Scaling (T = 0.5)

The v3 cancer model was initialized using v2 weights for faster convergence.

📂 Dataset

Dataset Source:

CT Kidney Dataset — Normal, Cyst, Tumor, Stone

Total images: 12,446

Split	Normal	Cyst	Stone	Tumour	Total
Train	5,077	1,800	952	2,079	9,908
Val	1,089	386	204	446	2,125
Test	1,089	386	224	446	2,145
Total	7,255	2,572	1,380	2,971	12,446
🧼 Preprocessing Pipeline
Resize → 224×224 (Lanczos)
      ↓
CLAHE Enhancement (clipLimit=4.0)
      ↓
BGR → RGB conversion

Dataset split:

MD5 filename hashing
70% Train
15% Validation
15% Test

This ensures deterministic splits without random seeds.

🔬 Cancer Label Mapping (v3)
v2 Class	v3 Label
Tumour	Cancer
Normal	Not Cancer
Cyst	Not Cancer
Stone	Not Cancer
📁 Project Structure
kidney-stone-cnn
│
├── api
│   ├── unified_main.py
│   ├── unified_inference.py
│   ├── unified_report.py
│   ├── main.py
│   └── inference.py
│
├── checkpoints
│   ├── best_model.pth
│   ├── best_model_v2.pth
│   └── best_model_v3.pth
│
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   ├── 03_gradcam.ipynb
│   ├── 05_train_v2.ipynb
│   └── 06_cancer_detection.ipynb
│
├── src
├── scripts
├── data
├── reports
├── monitoring
│
├── nephroscan_unified.html
├── nephroscan.html
└── requirements.txt
🖥 Dashboard Features

Dark UI (navy / teal design)

Drag-and-drop CT upload

3 model result cards with confidence bars

Risk banner (NORMAL / LOW / MEDIUM / HIGH)

Grad-CAM heatmap visualization

Prediction history table

Downloadable clinical PDF report

⚠️ Known Limitations
Limitation	Description
Research Only	Not validated for clinical deployment
CT Only	Model trained only on CT scans
No Patient Split	Dataset lacks patient IDs
No API Auth	API should not be exposed publicly
Single Organ	Only kidney analysis supported
📜 License

Dataset: CC BY 4.0
Code & Model Weights: Internal Research Project

⚠️ Disclaimer

This project is intended for research and portfolio purposes only.

It is not approved for clinical use and must not be used to make medical decisions.

All AI predictions should be reviewed by qualified healthcare professionals.
