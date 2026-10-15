from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ============================================================
# VISIONSHIELD - FASTAPI BACKEND
# ============================================================

app = FastAPI(
    title="VisionShield API",
    description="VisionShield PPE monitoring API",
    version="1.0.0",
)


# ============================================================
# LATEST DETECTION
# ============================================================

latest_detection: dict[str, Any] = {
    "timestamp": None,
    "workers": [],
    "worker_count": 0,
    "safe_count": 0,
    "unsafe_count": 0,
    "global_status": "NO WORKER",
}


# ============================================================
# LATEST VIOLATION
# ============================================================

latest_violation: dict[str, Any] = {
    "timestamp": None,
    "worker_id": None,
    "violation": False,
    "duration_seconds": 0,
    "missing_ppe": [],
    "detected_ppe": [],
    "message": None,
}


# ============================================================
# DETECTION MODEL
# ============================================================

class DetectionUpdate(BaseModel):

    timestamp: float | None = None

    workers: list[dict[str, Any]] = Field(
        default_factory=list
    )

    worker_count: int = 0

    safe_count: int = 0

    unsafe_count: int = 0

    global_status: str = "NO WORKER"


# ============================================================
# VIOLATION MODEL
# ============================================================

class ViolationUpdate(BaseModel):

    timestamp: float | None = None

    worker_id: int | str | None = None

    violation: bool = True

    duration_seconds: float = 0.0

    worker_count: int = 0

    unsafe_count: int = 0

    missing_ppe: list[str] = Field(
        default_factory=list
    )

    detected_ppe: list[str] = Field(
        default_factory=list
    )

    message: str = "PPE violation detected"

    # Optional worker information
    workers: list[dict[str, Any]] = Field(
        default_factory=list
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "VisionShield",
        "status": "online",
        "api": "running",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "VisionShield API",
    }


# ============================================================
# RECEIVE NORMAL DETECTIONS
# ============================================================

@app.post("/api/detections")
def receive_detection(
    detection: DetectionUpdate
):

    global latest_detection

    data = detection.model_dump()

    if data["timestamp"] is None:

        data["timestamp"] = (
            datetime.now(
                timezone.utc
            ).timestamp()
        )

    latest_detection = data

    return {
        "success": True,
        "message": "Detection received",
        "worker_count": data["worker_count"],
        "global_status": data["global_status"],
    }


# ============================================================
# RECEIVE PPE VIOLATION
# ============================================================

@app.post("/api/violations")
def receive_violation(
    violation: ViolationUpdate
):

    global latest_violation

    data = violation.model_dump()

    if data["timestamp"] is None:

        data["timestamp"] = (
            datetime.now(
                timezone.utc
            ).timestamp()
        )

    latest_violation = data

    print()
    print("=" * 70)
    print("VISIONSHIELD - PPE VIOLATION RECEIVED")
    print("=" * 70)

    print(
        f"Worker ID: {data['worker_id']}"
    )

    print(
        f"Duration: "
        f"{data['duration_seconds']:.2f} seconds"
    )

    print(
        f"Missing PPE: "
        f"{data['missing_ppe']}"
    )

    print(
        f"Detected PPE: "
        f"{data['detected_ppe']}"
    )

    print(
        f"Message: "
        f"{data['message']}"
    )

    print("=" * 70)

    return {
        "success": True,
        "message": "Violation received",
        "worker_id": data["worker_id"],
        "duration_seconds": data["duration_seconds"],
    }


# ============================================================
# CURRENT DETECTION STATUS
# ============================================================

@app.get("/api/status")
def get_status():

    return latest_detection


# ============================================================
# CURRENT VIOLATION
# ============================================================

@app.get("/api/violations/latest")
def get_latest_violation():

    return latest_violation


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )