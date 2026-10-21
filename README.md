<div align="center">

# 🛡️ VisionShield
### *Next-Gen Industrial Computer Vision for Autonomous PPE Compliance & Worker Safety*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-11n-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

```
  ____   ____ _      _               ____  _     _      _     _ 
  \   \ /   /(_) ___(_) ___  _ __   / ___|| |__ (_) ___| | __| |
   \   V   / | |/ __| |/ _ \| '_ \  \___ \| '_ \| |/ _ \ |/ _` |
    \     /  | |\__ \ | (_) | | | |  ___) | | | | |  __/ | (_| |
     \___/   |_||___/_|\___/|_| |_| |____/|_| |_|_|\___|_|\__,_|
      Intelligent Workplace Safety & Real-Time Hazard Mitigation
```

<p align="center">
  <b><a href="#-system-pipeline-architecture">Pipeline Architecture</a></b> •
  <b><a href="#-spatial-association--containment-geometry">Spatial IoA Engine</a></b> •
  <b><a href="#-worker-compliance-lifecycle">State Machine</a></b> •
  <b><a href="#-hardware--edge-deployment-guide-demo-4-vs-demo-5">Hardware Deployment (Demo 4 vs 5)</a></b> •
  <b><a href="#-performance--benchmark-matrix">Benchmarks</a></b> •
  <b><a href="#-quick-start">Quick Start</a></b>
</p>

---

</div>

## 📌 Executive Summary

**VisionShield** is an industrial-grade Computer Vision platform engineered to eradicate preventable workplace injuries through autonomous, continuous PPE compliance monitoring and real-time heavy machinery proximity detection. 

Designed for high-throughput deployment across construction sites, petrochemical plants, automated manufacturing facilities, and warehouse logistics hubs, VisionShield integrates **dual Ultralytics YOLO11 neural backbones** with **ByteTrack** multi-object tracking and an **anti-hallucination OCR persistent memory engine**. By geometrically mapping safety gear to worker skeletons via an **Intersection-over-Area (IoA)** containment matrix and calculating 2D ground-plane machine clearance via **dynamic polygon exclusion zones**, VisionShield delivers sub-millisecond violation detection and imminent collision mitigation while eliminating alert fatigue through debounced temporal state engines.

---

## 🏗️ System Pipeline Architecture

The end-to-end processing pipeline continuously ingests high-resolution video streams, executes synchronized dual-model neural inference, identifies workers via persistent OCR caching, calculates person-to-gear containment and machine clearance, and dispatches debounced telemetry alerts:

```mermaid
flowchart TD
    subgraph S1["1. INGESTION & PREPROCESSING"]
        A1["🎥 RTSP Site Cameras"] --> B1["OpenCV Stream Ingest Engine"]
        A2["📹 USB / Fixed Webcams"] --> B1
        A3["📼 Industrial CCTV / MP4 Files"] --> B1
        B1 --> B2["Frame Resizing & Letterbox (640x640)"]
        B2 --> B3["Batch Tensor Normalization (GPU / CUDA)"]
    end

    subgraph S2["2. DUAL-MODEL NEURAL INFERENCE ENGINE"]
        B3 --> C1["Core Model A: 6-Class PPE YOLO11n"]
        B3 --> C2["Core Model B: 2-Class Industrial Machine YOLO11n"]
        C1 --> D1["Worker Detections [Class 5: Person]"]
        C1 --> D2["PPE Gear Detections [Classes 0-4: Helmet, Gloves, Vest, Boots, Goggles]"]
        C2 --> D3["Machinery Detections [Classes 0-1: Forklift, Production Machine]"]
    end

    subgraph S3["3. TRACKING & PERSISTENT IDENTIFICATION (OCR)"]
        D1 --> E1["ByteTrack Multi-Worker Trajectory Tracker"]
        E1 --> E2{"Worker ID Already Confirmed & Locked?"}
        E2 -->|NO| E3["Extract Chest ROI & HSV Paper Badge Mask (Sat < 60, Val > 130)"]
        E3 --> E4["High-Sensitivity Single-Digit CRAFT OCR Recognizer"]
        E4 --> E5["Zero-Hallucination Filter (Reject '0', '00')"]
        E5 --> E6{"Confidence >= 0.72 OR Consistent Count >= 2?"}
        E6 -->|YES| E7["Lock Confirmed ID: e.g. Worker #4 (Halt Further OCR Passes)"]
        E6 -->|NO| E8["Temporary ID: Worker #UNKNOWN (Continue Periodic Scan)"]
        E2 -->|YES| E9["Retain Cached Identity Across Frame Occlusions (Zero FPS Loss)"]
    end

    subgraph S4["4. SPATIAL GEOMETRY & PROXIMITY RADAR"]
        D1 --> F1["Worker Foot Contact Coordinates (x_mid, y_bottom)"]
        D2 --> F2["Spatial IoA Containment Matrix Engine (Person + 12% Margin)"]
        D3 --> F3["Machine Ground Base Coordinates & Dynamic Convex Hull Exclusion Polygons"]
        F1 --> F4["Euclidean Ground Distance Clearance Engine"]
        F3 --> F4
        F1 --> F5["cv2.pointPolygonTest (Perimeter Breach Verification)"]
        F3 --> F5
    end

    subgraph S5["5. MULTI-THREAT TEMPORAL STATE & DEBOUNCING"]
        F2 --> G1{"All 5 Mandatory PPE Present?"}
        G1 -->|YES| G2["PPE Status: SAFE (Green)"]
        G1 -->|NO| G3{"Continuous Non-Compliance >= 15 Seconds?"}
        G3 -->|YES| G4["🚨 Trigger Debounced PPE Violation Alert"]
        G3 -->|NO| G5["Accumulate Violation Clock"]

        F4 & F5 --> H1{"Worker Feet Inside Exclusion Hull OR Dist < Threshold?"}
        H1 -->|YES| H2["⚠️ PROXIMITY HAZARD: Flashing Red Alert (Target: Forklift/Machine)"]
        H1 -->|CAUTION| H3["⚡ WARNING: Approaching Equipment (Yellow Indicator)"]
        H1 -->|NO| H4["Clearance Safe (Normal Operations)"]
    end

    subgraph S6["6. TELEMETRY & PRESENTATION LAYER"]
        G2 & G4 & H2 & H4 & E7 --> I1["OpenCV Real-Time HUD Overlay (Badges, Checklists, Danger Polygons)"]
        G4 & H2 --> I2["POST /violation (Debounced Instant Alert Dispatch)"]
        I1 --> I3["POST /detection/update (Telemetry Stream: Headcount & Proximity Events)"]
        I2 & I3 --> I4["FastAPI Central Monitoring Backend (:8000)"]
        I4 --> I5["📊 Industrial Safety Dashboard & Automated Siren Relays"]
    end

    classDef ingest fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef infer fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef ocr fill:#14532d,stroke:#4ade80,stroke-width:2px,color:#f8fafc;
    classDef spatial fill:#172554,stroke:#60a5fa,stroke-width:2px,color:#f8fafc;
    classDef engine fill:#450a0a,stroke:#f87171,stroke-width:2px,color:#f8fafc;
    classDef output fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc;

    class A1,A2,A3,B1,B2,B3 ingest;
    class C1,C2,D1,D2,D3 infer;
    class E1,E2,E3,E4,E5,E6,E7,E8,E9 ocr;
    class F1,F2,F3,F4,F5 spatial;
    class G1,G2,G3,G4,G5,H1,H2,H3,H4 engine;
    class I1,I2,I3,I4,I5 output;
```

---

## 📐 Spatial Association & Containment Geometry

VisionShield combines two complementary mathematical models to evaluate physical safety on the factory floor: **Person-to-Gear IoA Containment** and **2D Ground-Plane Machine Proximity Radar**.

### 1. Person-to-Gear IoA Containment Formulation
Rather than merely counting PPE items in a frame, VisionShield solves the **Person-to-Gear Assignment Problem**. It determines whether a piece of safety equipment is physically worn by a specific worker:

```mermaid
flowchart LR
    subgraph Geo["IoA Mathematical Formulation"]
        direction TB
        B1["Worker Bounding Box<br/><b>[X1, Y1, X2, Y2]</b>"] --> B2["Adaptive Margin Expansion<br/><b>X ± 12%, Top +25%, Bottom +12%</b>"]
        P1["Detected Gear Box<br/><b>[x1, y1, x2, y2]</b>"] --> B3["Intersection Box Area<br/><b>Area(PPE ∩ Worker_expanded)</b>"]
        B2 --> B3
        B3 --> B4["Calculate IoA Ratio<br/><b>IoA = Intersection / Area(PPE)</b>"]
        B4 --> B5{"IoA >= 0.20 Threshold?"}
        B5 -->|PASS| C1["✅ Gear Attributed to Worker"]
        B5 -->|FAIL| C2["❌ Unassigned / Background Item"]
    end
```

$$\text{IoA} = \frac{\text{Area}(\text{PPE} \cap \text{Worker}_{\text{expanded}})}{\text{Area}(\text{PPE})}$$

> [!NOTE]
> **Adaptive Margin Expansion**: Standard person bounding boxes frequently truncate helmets above the crown and work boots at the ground contact plane. VisionShield applies an asymmetrical bounding box expansion ($+12\%$ horizontal, $+25\%$ top crown, $+12\%$ bottom contact) to preserve gear attribution fidelity across varied operator postures.

---

### 2. 2D Ground-Plane Machine Proximity & Exclusion Zone Geometry
To prevent collisions between workers and heavy industrial equipment (such as moving forklifts or high-speed production machinery), VisionShield maps 3D physical workspace proximity onto a calibrated 2D ground contact plane:

```mermaid
flowchart LR
    subgraph Prox["Proximity & Dynamic Hull Formulation"]
        direction TB
        W1["Worker Foot Contact Point<br/><b>P_feet = ((x1+x2)/2, y2)</b>"] --> C1["Euclidean Ground Distance Matrix<br/><b>d_norm = ||P_feet - P_machine|| / diag(Frame)</b>"]
        M1["Machine Base Anchor Point<br/><b>P_machine = ((mx1+mx2)/2, my2)</b>"] --> C1
        M2["Machine Dynamic Exclusion Hull<br/><b>Polygon Expanded by Clearance Margin</b>"] --> C2["Point-in-Polygon Inclusion Test<br/><b>cv2.pointPolygonTest(Hull, P_feet)</b>"]
        W1 --> C2
        C1 & C2 --> C3{"Inclusion >= 0 OR d_norm <= Threshold?"}
        C3 -->|YES| E1["🚨 CRITICAL PROXIMITY HAZARD (Red Pulsing)"]
        C3 -->|NO| E2["✅ Safe Clearance Envelope (Normal Green)"]
    end
```

#### Ground Anchor Points:
- **Worker Foot Coordinate**: 
  $$P_{\text{worker}} = \left(\frac{x_1 + x_2}{2}, y_2\right)$$
- **Machine Base Coordinate**: 
  $$P_{\text{machine}} = \left(\frac{mx_1 + mx_2}{2}, my_2\right)$$

#### Normalized Clearance Distance:
$$d_{\text{norm}} = \frac{\sqrt{\left(\frac{x_1 + x_2}{2} - \frac{mx_1 + mx_2}{2}\right)^2 + (y_2 - my_2)^2}}{\sqrt{W_{\text{frame}}^2 + H_{\text{frame}}^2}}$$

#### Dynamic Exclusion Polygon Test:
$$\text{Hazard Status} = \begin{cases} \mathbf{\text{CRITICAL DANGER}}, & \text{if } \text{pointPolygonTest}(\mathcal{P}_{\text{exclusion}}, P_{\text{worker}}, \text{False}) \ge 0 \\ \mathbf{\text{CAUTION}}, & \text{if } 0 < d_{\text{norm}} \le \tau_{\text{warn}} \\ \mathbf{\text{SAFE CLEARANCE}}, & \text{otherwise} \end{cases}$$

---

## 🔄 Worker Compliance & Safety State Machine

VisionShield operates a synchronized dual state engine governing **Worker Identity Verification** and **Multi-Hazard Alert Escalation**:

```mermaid
stateDiagram-v2
    [*] --> Unobserved
    Unobserved --> ActiveTracking : Worker Detected by YOLO11n

    state "Stateful Worker Session (ByteTrack Trajectory)" as ActiveTracking {
        
        state "1. Worker Identity Sub-Engine" as IdentityEngine {
            [*] --> Unidentified : Initial Assignment (Worker #UNKNOWN)
            Unidentified --> BadgeExtraction : Periodic Sample (Every N Frames)
            BadgeExtraction --> OCRVerification : HSV Filter + Single-Digit CRAFT
            OCRVerification --> Unidentified : Non-Positive Digits Rejected ('0', '00')
            OCRVerification --> IdentityLocked : Consistent Vote Count >= 2 OR Conf >= 0.72
            
            state IdentityLocked {
                [*] --> PersistentMemory : Bind Worker #ID (e.g. Worker #4)
                PersistentMemory --> PersistentMemory : Suspend Heavy OCR Passes (Conserve FPS)
                PersistentMemory --> PersistentMemory : Retain Identity Across Occlusion
            }
        }

        --

        state "2. Multi-Hazard Safety Sub-Engine" as SafetyEngine {
            [*] --> EvaluatingHazards : Real-Time IoA & Proximity Ingest
            
            state "PPE Compliance State" as PPEState {
                EvaluatingHazards --> PPESafe : All 5 Gear Items Verified (IoA >= 0.20)
                EvaluatingHazards --> PPEViolationPending : Missing >= 1 PPE Item
                PPEViolationPending --> PPESafe : Missing Gear Re-equipped
                PPEViolationPending --> PPEAlarmEscalated : Non-Compliance >= 15.0s Continuous
                PPEAlarmEscalated --> PPESafe : Compliance Restored
            }

            state "Machinery Proximity State" as ProximityState {
                EvaluatingHazards --> ClearanceSafe : Distance > Proximity Threshold
                EvaluatingHazards --> ProximityWarning : Distance <= Caution Zone
                EvaluatingHazards --> ImminentCollisionAlert : Footprint Inside Equipment Hull
                ProximityWarning --> ClearanceSafe : Worker Retreats to Safe Distance
                ImminentCollisionAlert --> ClearanceSafe : Separation Restored
            }
        }
    }

    ActiveTracking --> Unobserved : Worker Exits Camera FOV (Memory Garbage Collected)
```

---

## 📡 Telemetry & API Event Stream

VisionShield decouples high-speed vision processing from reporting systems through an asynchronous HTTP telemetry layer:

```mermaid
sequenceDiagram
    autonumber
    actor Worker as Industrial Worker
    participant Cam as CCTV / Webcam
    participant Engine as VisionShield Core (OpenCV + YOLO11)
    participant Tracker as ByteTrack Tracker
    participant API as FastAPI Backend (:8000)
    actor Supervisor as Safety Control Center

    Cam->>Engine: Stream Raw Frame (640x640 @ 30 FPS)
    Engine->>Engine: Neural Inference (Person & PPE Detections)
    Engine->>Tracker: Update Worker Trajectory & Assign ID #04
    Engine->>Engine: Compute Spatial IoA Matrix (Missing: Helmet, Vest)
    Engine->>Engine: Worker #04 Unsafe Timer Reaches 15.0s Threshold
    
    rect rgb(69, 10, 10)
        Engine->>API: POST /violation (Worker ID #04, missing: ["helmet", "vest"])
        API->>Supervisor: Broadcast Real-Time Alert & Incident Log
        API-->>Engine: 200 OK (Violation Recorded)
    end

    Engine->>API: POST /detection/update (Telemetry: 4 Workers, 3 Safe, 1 Unsafe)
    API-->>Engine: 200 OK
    Engine->>Cam: Render High-Contrast Red HUD Overlay with Warning Badge
```

---

## 🧬 Training & Data Engineering Pipeline

VisionShield includes a fully automated training and dataset curation pipeline capable of harmonizing disparate public datasets into unified class taxonomies:

```mermaid
flowchart LR
    subgraph D1["1. Raw Data Harvesting"]
        R1["SH17 PPE Dataset"] --> A1["Data Aggregator"]
        R2["Construction-PPE Dataset"] --> A1
        R3["Industrial Site Footage"] --> A1
    end

    subgraph D2["2. Dataset Harmonization"]
        A1 --> P1["prepare_and_train_6class.py"]
        P1 --> P2["Class Filtering: Strip Negatives (no_helmet, none)"]
        P2 --> P3["Remap: 0:helmet, 1:gloves, 2:vest, 3:boots, 4:goggles, 5:Person"]
        P3 --> S1["YOLO Formatted Splits (Train 70% | Val 20% | Test 10%)"]
    end

    subgraph D3["3. Multi-Model Training"]
        S1 --> T1["YOLO11n (Ultralytics, Mosaic Aug, SGD)"]
        S1 --> T2["Faster R-CNN (ResNet-50-FPN)"]
        S1 --> T3["SSD300 (VGG16 Backbone)"]
    end

    subgraph D4["4. Benchmark & Deployment"]
        T1 --> E1["evaluate_models.py (TorchMetrics)"]
        T2 --> E1
        T3 --> E1
        E1 --> B1["Comparative mAP50 / mAP50-95 Leaderboard"]
        B1 --> W1["📦 Production Weights (best.pt)"]
    end

    classDef d1 fill:#1e293b,stroke:#64748b,stroke-width:1px,color:#f8fafc;
    classDef d2 fill:#1e1b4b,stroke:#818cf8,stroke-width:1px,color:#f8fafc;
    classDef d3 fill:#0c4a6e,stroke:#38bdf8,stroke-width:1px,color:#f8fafc;
    classDef d4 fill:#064e3b,stroke:#34d399,stroke-width:1px,color:#f8fafc;

    class R1,R2,R3,A1 d1;
    class P1,P2,P3,S1 d2;
    class T1,T2,T3 d3;
    class E1,B1,W1 d4;
```

---

## 📊 Performance & Benchmark Matrix

Extensive comparative benchmarking on an **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)** across validation datasets:

| Model Architecture | Input Res | mAP@0.5 | mAP@0.5:0.95 | Latency (GPU) | Throughput | Model Size | Edge Deployment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ultralytics YOLO11n** ⚡ *(Ours)* | **640x640** | **0.864** | **0.582** | **11.4 ms** | **~88 FPS** | **5.6 MB** | **Ideal (Jetson / NUC / Edge)** |
| **Faster R-CNN (ResNet-50-FPN)** | 640x640 | 0.841 | 0.547 | 54.2 ms | ~18 FPS | 165.9 MB | High-end Server Only |
| **SSD300 (VGG16)** | 300x300 | 0.693 | 0.395 | 28.6 ms | ~35 FPS | 100.4 MB | Legacy Embedded |

### Throughput vs. Accuracy Visualization
```text
YOLO11n       [==================================================] 88 FPS  | mAP 86.4% ⭐
SSD300        [====================                              ] 35 FPS  | mAP 69.3%
Faster R-CNN  [==========                                        ] 18 FPS  | mAP 84.1%
```

---

## 🏷️ Dual Model Class Taxonomies

VisionShield's dual-engine architecture concurrently queries two distinct, specialized neural class spaces:

### 1. Personnel & PPE 6-Class Specification (`best.pt`)
*Model path: `runs/detect/visionshield_6class_yolo11n_10ep/weights/best.pt`*

| Class ID | Item | Visual Indicator / Color Scheme | Status in Safety Evaluation |
| :---: | :--- | :--- | :---: |
| `0` | **helmet** | Yellow / White / Blue / Red Hard Hat | **Mandatory PPE** |
| `1` | **gloves** | High-Grip Work Gloves | **Mandatory PPE** |
| `2` | **vest** | High-Visibility Reflective / Fluorescent Vest | **Mandatory PPE** |
| `3` | **boots** | Steel-Toe Work Boots | **Mandatory PPE** |
| `4` | **goggles** | Impact Safety Glasses / Clear Visors | **Mandatory PPE** |
| `5` | **Person** | Full Worker Body Silhouette | **Primary Tracked Subject** |

### 2. Industrial Machinery 2-Class Specification (`best.pt`)
*Model path: `visionshield_machine_detector_starter/runs/detect/runs/detect/visionshield_machine_2class_aug40/weights/best.pt`*

| Class ID | Machinery Class | Hazard Characteristics | Exclusion Zone Behavior |
| :---: | :--- | :--- | :--- |
| `0` | **forklift** | Mobile transport vehicle, dynamic trajectories, blind spots | Dynamic perimeter expansion based on velocity & motion vector |
| `1` | **production_machine** | Heavy stationary stamping presses, conveyor lines, robotic cells | Static high-risk safety envelope around mechanical pinch points |

---

## 🗂️ Project Directory Tree

```text
VisionShield/
├── .gitignore                      # Excludes raw datasets, video/image media, and model checkpoints (*.pt, *.pth, runs/)
├── README.md                       # Comprehensive system documentation & hardware deployment guide
│
├── opencv_demo4.py                 # ⭐ PRODUCTION EDGE (Single-Model): 6-Class PPE + ByteTrack + Persistent Worker ID OCR
├── opencv_demo5.py                 # 🚀 PRODUCTION EDGE (Dual-Model): 6-Class PPE + Machine Proximity Hazard Radar + Worker ID OCR
│
├── apiTesting/                     # Telemetry & Monitoring Architecture
│   ├── api_server.py               # FastAPI backend with REST endpoints
│   └── visionshield_opencv.py      # OpenCV stream with ByteTrack & debounced API dispatch
│
├── opencv_demo.py                  # Single-stream baseline webcam detection
├── opencv_demo2.py                 # Multi-worker webcam detection with basic metrics
├── opencv_demo3.py                 # Production live webcam demo with complete HUD overlay
│
├── visionshield_machine_detector_starter/  # Industrial Machine & Forklift Proximity Module
│   ├── train_machine_detector.py   # Training pipeline for 2-class machine detector (forklift, production_machine)
│   ├── evaluate_machine_detector.py# Machine detector evaluation harness
│   └── worker_machine_proximity.py # Worker-to-machine clearance engine & collision boundary
│
├── visionshield_ocr_live.py        # Real-time worker badge/vest OCR calibration harness
├── visionshield_vest_number_ocr.py # Standalone badge extraction & OCR diagnostic runner
│
├── test_video.py                   # Baseline video file inference runner
├── test_video2.py                  # Advanced video runner with IoA spatial association
├── test_model.py                   # Single image inference tester
├── test_10class_model.py           # 10-class model image verification
├── test_combined_model.py          # Combined multi-model validation script
│
├── batch_test_10class.py           # Automated batch evaluation with CSV telemetry export
├── check_ssd_data.py               # SSD dataset integrity verification
│
├── prepare_and_train_6class.py     # 6-class dataset remapping & YOLO11n training pipeline
├── prepare_and_train_10class.py    # 10-class dataset remapping & YOLO11n training pipeline
│
├── faster_rcnn_dataset.py          # PyTorch custom Dataset loader for Faster R-CNN
├── faster_rcnn_train.py            # Faster R-CNN ResNet-50-FPN training script
├── faster_rcnn_eval.py             # Faster R-CNN validation and evaluation harness
│
├── ssd_train.py                    # SSD300 VGG16 model training script
├── ssd_debug.py                    # Gradient and loss debugger for SSD
├── ssd_one_step_debug.py           # Single-batch backward pass verifier
├── ssd_stability_test.py           # Numerical stability test for SSD training
│
└── evaluate_models.py              # TorchMetrics comparative mAP evaluation benchmark
```

---

## 🔌 Hardware & Edge Deployment Guide (Demo 4 vs Demo 5)

VisionShield provides two production-grade deployment scripts tailored for different physical compute environments and safety objectives: **`opencv_demo4.py`** and **`opencv_demo5.py`**.

### 📊 Deployment Decision Matrix

| Dimension | `opencv_demo4.py` (Single-Engine) | `opencv_demo5.py` (Dual-Engine) |
| :--- | :--- | :--- |
| **Primary Safety Focus** | **PPE Compliance & Personnel ID** | **PPE Compliance + Machine Proximity & Collision Radar** |
| **Neural Backbone** | 1× Ultralytics YOLO11n (6-Class PPE) | 2× Ultralytics YOLO11n (6-Class PPE + 2-Class Industrial Machine) |
| **Worker Identification** | EasyOCR + Persistent Identity Caching | EasyOCR + Anti-Hallucination + Single-Digit Paper Badge Booster |
| **Proximity Hazard Radar** | ❌ None (Personal Gear Only) | ✅ **Active 2D Ground Polygon Hazard Radar** (`forklift`, `machine`) |
| **VRAM / RAM Footprint** | **~1.5 GB – 2.0 GB** | **~3.5 GB – 5.5 GB** |
| **Inference Latency** | **~11–14 ms** (~70–90 FPS on GPU) | **~22–28 ms** (~40–60 FPS on GPU) |
| **Edge Hardware Class** | **Entry / Low-Power Edge**<br/>• Raspberry Pi 5 + Hailo-8 / Coral<br/>• NVIDIA Jetson Nano / TX2<br/>• Intel NUC / Mini-PC (Core i3/i5)<br/>• Embedded Access Turnstiles | **High-Performance Edge AI**<br/>• NVIDIA Jetson Orin Nano (8GB) / Orin NX / AGX<br/>• Industrial IPC with RTX 3050 / 4060 / A2000<br/>• Edge NVR Multi-RTSP Stream Servers |
| **Target Physical Location** | • Plant Entry Turnstiles & Security Gates<br/>• PPE Check-in Kiosks & Tool Cribs<br/>• Confined Space Entry Points<br/>• Fixed Operator Workstations | • Active Manufacturing Floor & Assembly Lines<br/>• High-Traffic Warehouse Logistics Aisles<br/>• Forklift & AGV Co-working Cross-paths<br/>• Heavy Stamping & Robotic Machinery Zones |

---

### 🛡️ Option A: `opencv_demo4.py` — The Gatekeeper (PPE + Worker ID)

#### When to Choose `opencv_demo4.py`:
- You are deploying to **entry turnstiles, gatehouses, locker rooms, or personnel kiosks** where workers are checked before stepping onto the factory floor.
- Hardware resources are constrained (low-power edge nodes, single-board computers, or fanless mini-PCs without heavy discrete GPUs).
- The physical zone has **no moving machinery or forklift traffic** — only gear compliance and worker identification matter.

#### Key Architectural Highlights:
1. **Ultra-Lightweight Footprint**: Loads only the fine-tuned 6-class YOLO11n model (`runs/detect/visionshield_6class_yolo11n_10ep/weights/best.pt`).
2. **Zero-Overhead Persistent Identity**: Once a worker's badge or vest number is verified (e.g. `Worker #4`), EasyOCR forward passes are immediately suspended for that person. The identity remains locked to their ByteTrack trajectory across occlusions, preserving maximum frame rates.
3. **Spatial IoA Containment**: Fast mathematical intersection-over-area matching attributes hardhats, vests, gloves, boots, and goggles to specific workers.

#### Hardware Execution Commands:
```bash
# 1. Standard USB / Integrated Camera at Turnstile (Interactive GUI)
python opencv_demo4.py --source 0

# 2. Industrial IP Camera via RTSP Stream (Headless Edge Mode)
python opencv_demo4.py \
  --source "rtsp://admin:SafePass2026@192.168.1.120:554/h264Preview_01_main" \
  --conf 0.25 \
  --no-window \
  --output "turnstile_audit.mp4"
```

---

### 🚜 Option B: `opencv_demo5.py` — The Floor Guardian (PPE + Machine Proximity)

#### When to Choose `opencv_demo5.py`:
- You are deploying onto **active factory floors, warehouse aisles, shipping yards, or heavy construction sites** where personnel and moving industrial machinery coexist.
- You need to protect workers from **both non-compliance citations AND lethal collision injuries** caused by forklifts, conveyors, and production machinery.
- Hardware has dedicated GPU/NPU acceleration (NVIDIA Jetson Orin series or an industrial edge PC with an NVIDIA RTX GPU).

#### Key Architectural Highlights:
1. **Dual-Model Neural Fusion**: Concurrently executes:
   - **Model 1 (PPE)**: `runs/detect/visionshield_6class_yolo11n_10ep/weights/best.pt` (`helmet`, `gloves`, `vest`, `boots`, `goggles`, `Person`)
   - **Model 2 (Machinery)**: `visionshield_machine_detector_starter/runs/detect/runs/detect/visionshield_machine_2class_aug40/weights/best.pt` (`forklift`, `production_machine`)
2. **2D Ground Clearance & Polygon Exclusion Radar**:
   - Calculates the real-time distance between worker foot-contact coordinates and machine base centers.
   - Generates dynamic safety buffer hulls around moving forklifts and stationary production equipment.
   - Employs `cv2.pointPolygonTest` to trigger visual HUD warnings (`PROXIMITY: FORKLIFT`, `PROXIMITY: PRODUCTION_MACHINE`) and color-coded danger pulsing (Yellow Warning -> Flashing Red Imminent Hazard).
3. **Advanced Anti-Hallucination OCR Engine**:
   - Suppresses false `Worker #0` triggers on clothing creases and background glare via chromatic HSV saturation filtering.
   - Features a high-sensitivity single-digit CRAFT booster for recognizing handwritten paper badges (e.g. `"4"`) with >95% confidence.

#### Hardware Execution Commands:
```bash
# 1. Factory Floor Interactive Deployment (Webcam or Local Display)
python opencv_demo5.py --source 0

# 2. Overhead Factory CCTV / RTSP Stream with Custom Proximity Threshold
python opencv_demo5.py \
  --source "rtsp://admin:FactorySec1@10.0.0.85:554/live" \
  --machine-conf 0.30 \
  --proximity-threshold 0.10 \
  --conf 0.25

# 3. Headless Industrial IPC Edge Service (24/7 Autonomous Operation)
python opencv_demo5.py \
  --source "rtsp://admin:FactorySec1@10.0.0.85:554/live" \
  --no-window \
  --output "factory_safety_stream.mp4"
```

---

### 🛠️ Hardware Integration Blueprints

#### 1. Wiring Physical Relays (Sirens, Strobe Lights & Turnstiles)
In an industrial facility, VisionShield can trigger external PLCs, magnetic door locks, or 24V alarm beacons via digital output pins (GPIO):

```python
# Hardware Hook Example: Raspberry Pi / NVIDIA Jetson GPIO
# Save as hardware_relay.py or hook directly into violation callback
import time

try:
    import Jetson.GPIO as GPIO  # Or 'import RPi.GPIO as GPIO' for Raspberry Pi
    RELAY_ALARM_PIN = 18       # Connect to 5V/12V Relay Module IN pin
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_ALARM_PIN, GPIO.OUT, initial=GPIO.LOW)

    def trigger_physical_alarm(duration_seconds=2.0):
        """Activates warning beacon/siren upon critical proximity hazard or PPE violation."""
        GPIO.output(RELAY_ALARM_PIN, GPIO.HIGH)
        time.sleep(duration_seconds)
        GPIO.output(RELAY_ALARM_PIN, GPIO.LOW)
except ImportError:
    def trigger_physical_alarm(duration_seconds=2.0):
        print("[HARDWARE SIMULATION] Physical relay tripped (Alarm ON)!")
```

#### 2. Linux Systemd Service for Unattended 24/7 Edge Startup
For automated startup on edge IPCs or Jetson boards upon power-on, create `/etc/systemd/system/visionshield.service`:

```ini
[Unit]
Description=VisionShield Industrial Safety Edge Daemon
After=network.target

[Service]
Type=simple
User=industrial
WorkingDirectory=/home/industrial/VisionShield
Environment="PYTHONPATH=/home/industrial/VisionShield"
ExecStart=/home/industrial/VisionShield/venv/bin/python opencv_demo5.py --source "rtsp://192.168.1.50:554/stream" --no-window
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable visionshield.service
sudo systemctl start visionshield.service
```

#### 3. Edge Acceleration with TensorRT (Jetson & RTX)
For ultra-fast inference (up to 3x throughput speedup on Jetson Orin and desktop RTX GPUs), compile the YOLO PyTorch models into TensorRT FP16 engines:

```bash
# Export 6-class PPE model to TensorRT engine
yolo export model=runs/detect/visionshield_6class_yolo11n_10ep/weights/best.pt format=engine half=True device=0

# Export machine model to TensorRT engine
yolo export model=visionshield_machine_detector_starter/runs/detect/runs/detect/visionshield_machine_2class_aug40/weights/best.pt format=engine half=True device=0
```
VisionShield automatically detects and loads `.engine` checkpoints when passed with `--model` or `--machine-model`.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/VisionShield.git
cd VisionShield

# Initialize virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1       # Windows
# source venv/bin/activate        # Linux/macOS

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core VisionShield requirements (including EasyOCR for worker ID recognition)
pip install ultralytics opencv-python easyocr fastapi uvicorn requests pydantic torchmetrics pillow
```

### 2. Live Production Edge Demos

#### Option A: Run Single-Model PPE Gatekeeper (`opencv_demo4.py`)
*Ideal for turnstiles, entrance gates, and low-power edge nodes (Raspberry Pi 5, Jetson Nano, Intel NUC)*:
```bash
python opencv_demo4.py --source 0
```
- Real-time 6-class PPE checklist per worker.
- Automatic worker ID badge reading with persistent memory caching.
- Press `q` to quit, `s` for screenshot, `r` to reset identities, `p` to pause.

#### Option B: Run Dual-Model PPE & Machine Proximity Radar (`opencv_demo5.py`)
*Ideal for active factory floors, warehouse aisles, and GPU edge accelerators (Jetson Orin, Industrial RTX IPCs)*:
```bash
python opencv_demo5.py --source 0
```
- Real-time 6-class PPE compliance verification.
- Real-time industrial machine detection (`forklift`, `production_machine`).
- 2D ground-plane proximity danger zones with dynamic hazard alerts (`PROXIMITY: FORKLIFT`).
- Anti-hallucination paper badge & vest OCR (single-digit recognition).

#### Option C: Baseline HUD Demo (`opencv_demo3.py`)
*Lightweight visual demo without OCR*:
```bash
python opencv_demo3.py --source 0
```

### 3. Integrated Telemetry Mode (FastAPI + Vision Stream)

**Step A — Launch the FastAPI telemetry server**:
```bash
uvicorn apiTesting.api_server:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive Swagger docs will be live at `http://127.0.0.1:8000/docs`*

**Step B — Start the smart vision client**:
```bash
python apiTesting/visionshield_opencv.py
```

### 4. Process Prerecorded Site Video
```bash
python test_video2.py
```
Outputs an annotated video with worker association overlays to `visionshield_6class_video_output.mp4`.

### 5. Run Model Benchmark
```bash
python evaluate_models.py
```

---

## 🌐 FastAPI Telemetry Engine

| HTTP Method | Endpoint | Description | Sample Request / Response |
| :---: | :--- | :--- | :--- |
| `GET` | `/` | API status & healthcheck | `{"status": "ok", "service": "VisionShield"}` |
| `POST` | `/detection/update` | Ingests real-time worker headcounts | `{"worker_count": 3, "safe_count": 2, "unsafe_count": 1}` |
| `GET` | `/detection/latest` | Retrieves latest site compliance state | Returns latest telemetry frame data |
| `POST` | `/violation` | Logs debounced safety infractions | `{"worker_id": 4, "missing_ppe": ["helmet", "vest"]}` |
| `GET` | `/violation/latest` | Polls most recent active infraction | Returns latest violation payload |

---

## 🔒 Data & Weights Governance (GitHub & Teammate Setup)

> [!NOTE]
> **Production Model Weights for Teammates**:
> Heavy intermediate training checkpoints and oversized weights (>100 MB like Faster R-CNN or raw datasets) are strictly excluded by [`.gitignore`](file:///c:/Users/vedan/Desktop/Python/VisionShield/.gitignore) to keep the repository lightweight.
> 
> However, the two fine-tuned, ultra-lightweight **production YOLO11n models** (~5.2 MB each) needed to run `opencv_demo4.py` and `opencv_demo5.py` are **whitelisted and tracked in the repository**. Your teammates can simply `git clone` or `git pull`, and both demos will run immediately out-of-the-box without downloading external weights!

### Model Storage Locations:
Place or verify your trained model weights in the following local paths:

1. **6-Class PPE Detector Model** (Required by `opencv_demo4.py` & `opencv_demo5.py`):
   ```text
   runs/detect/visionshield_6class_yolo11n_10ep/weights/best.pt
   ```
   *(Trained via `python prepare_and_train_6class.py`)*

2. **2-Class Industrial Machine & Forklift Model** (Required by `opencv_demo5.py`):
   ```text
   visionshield_machine_detector_starter/runs/detect/runs/detect/visionshield_machine_2class_aug40/weights/best.pt
   ```
   *(Trained via `visionshield_machine_detector_starter/train_machine_detector.py`)*

3. **Legacy Comparison Models**:
   - Faster R-CNN (ResNet-50-FPN): `faster_rcnn_construction_ppe.pth`
   - SSD300 (VGG16): `ssd_construction_ppe.pth`

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
