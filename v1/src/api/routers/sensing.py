"""
Sensing REST API — mobile / external app integration.

Lightweight HTTP endpoints that expose real-time and historical sensing data
(presence, motion, vitals, node status) so that mobile apps, detectors, and
other third-party consumers can pull the information over plain REST.

All endpoints are registered under  ``/api/v1/sensing``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user, require_auth
from src.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# ---------------------------------------------------------------------------
# In-memory stores (replace with DB / Redis for production)
# ---------------------------------------------------------------------------

# Sensing history ring-buffer (last 1 000 snapshots ≈ ~8 min at 500 ms tick)
_history: deque[Dict[str, Any]] = deque(maxlen=1_000)

# Active alerts
_alerts: List[Dict[str, Any]] = []

# Alert rules
_alert_rules: List[Dict[str, Any]] = []

# Runtime-overridable config
_sensing_config: Dict[str, Any] = {
    "presence_variance_threshold": 0.5,
    "motion_energy_threshold": 0.1,
    "tick_interval_seconds": 0.5,
    "window_seconds": 10.0,
    "source": "auto",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SensingFeatures(BaseModel):
    mean_rssi: float = Field(..., description="평균 RSSI (dBm)")
    variance: float = Field(..., description="RSSI 분산")
    std: float = Field(..., description="표준편차")
    motion_band_power: float = Field(..., description="모션 밴드 에너지 (0.5-3 Hz)")
    breathing_band_power: float = Field(..., description="호흡 밴드 에너지 (0.1-0.5 Hz)")
    dominant_freq_hz: float = Field(..., description="주 주파수 (Hz)")
    spectral_power: float = Field(0.0, description="총 스펙트럼 파워")
    range: float = Field(0.0, description="RSSI 범위")
    iqr: float = Field(0.0, description="사분위 범위")
    skewness: float = Field(0.0, description="비대칭도")
    kurtosis: float = Field(0.0, description="첨도")


class ClassificationResult(BaseModel):
    motion_level: str = Field(..., description="모션 레벨: absent | present_still | active")
    presence: bool = Field(..., description="사람 존재 여부")
    confidence: float = Field(..., description="신뢰도 (0.0 ~ 1.0)")


class SensingCurrentResponse(BaseModel):
    timestamp: float = Field(..., description="Unix timestamp")
    source: str = Field(..., description="데이터 소스: esp32 | windows_wifi | simulated 등")
    node_count: int = Field(..., description="연결된 노드 수")
    presence: bool = Field(..., description="사람 존재 여부")
    motion_level: str = Field(..., description="absent | present_still | active")
    confidence: float = Field(..., description="신뢰도")
    features: SensingFeatures
    classification: ClassificationResult


class VitalsResponse(BaseModel):
    timestamp: float
    breathing_rate_bpm: Optional[float] = Field(None, description="호흡수 (BPM)")
    heart_rate_bpm: Optional[float] = Field(None, description="추정 심박수 (BPM)")
    breathing_confidence: float = Field(0.0, description="호흡 추정 신뢰도")
    status: str = Field("unknown", description="normal | elevated | low | unknown")


class NodeInfo(BaseModel):
    node_id: int
    rssi_dbm: float = 0.0
    position: List[float] = Field(default_factory=list)
    online: bool = False
    mean_amplitude: float = 0.0
    subcarrier_count: int = 0
    freq_mhz: int = 0
    source_addr: str = ""
    last_seen: float = 0.0


class AlertRule(BaseModel):
    name: str = Field(..., description="알림 규칙 이름")
    condition: str = Field(..., description="조건: presence_detected | motion_active | breathing_abnormal")
    threshold: Optional[float] = Field(None, description="임계값 (선택)")
    enabled: bool = Field(True, description="활성 여부")


class AlertEvent(BaseModel):
    id: str
    rule_name: str
    triggered_at: float
    condition: str
    value: float
    message: str


class SensingConfigResponse(BaseModel):
    presence_variance_threshold: float
    motion_energy_threshold: float
    tick_interval_seconds: float
    window_seconds: float
    source: str


class SensingConfigUpdate(BaseModel):
    presence_variance_threshold: Optional[float] = Field(None, ge=0.01, le=10.0)
    motion_energy_threshold: Optional[float] = Field(None, ge=0.01, le=5.0)
    tick_interval_seconds: Optional[float] = Field(None, ge=0.1, le=5.0)
    window_seconds: Optional[float] = Field(None, ge=1.0, le=120.0)
    source: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_latest_sensing(request: Request) -> Optional[Dict[str, Any]]:
    """
    Pull the latest sensing snapshot.

    Prefers data cached by the WebSocket server tick loop (stored in app
    state) so the REST API stays in sync with WebSocket clients.
    Falls back to the in-memory history ring-buffer.
    """
    # Try app state first (set by ws_server or by a middleware)
    latest = getattr(request.app.state, "latest_sensing", None)
    if latest:
        return latest

    # Fallback: most recent history entry
    if _history:
        return _history[-1]

    return None


def _check_alert_rules(snapshot: Dict[str, Any]) -> None:
    """Evaluate alert rules against the latest snapshot and fire events."""
    classification = snapshot.get("classification", {})
    features = snapshot.get("features", {})

    for rule in _alert_rules:
        if not rule.get("enabled", True):
            continue

        condition = rule.get("condition", "")
        triggered = False
        value = 0.0
        msg = ""

        if condition == "presence_detected" and classification.get("presence"):
            triggered = True
            value = classification.get("confidence", 0)
            msg = f"사람 감지됨 (신뢰도 {value:.0%})"

        elif condition == "motion_active" and classification.get("motion_level") == "active":
            triggered = True
            value = features.get("motion_band_power", 0)
            msg = f"움직임 감지됨 (모션 에너지 {value:.4f})"

        elif condition == "breathing_abnormal":
            bp = features.get("breathing_band_power", 0)
            threshold = rule.get("threshold", 0.3)
            if bp > threshold:
                triggered = True
                value = bp
                msg = f"호흡 이상 감지 (에너지 {bp:.4f} > 임계값 {threshold})"

        if triggered:
            event = {
                "id": str(uuid.uuid4()),
                "rule_name": rule.get("name", "unknown"),
                "triggered_at": time.time(),
                "condition": condition,
                "value": value,
                "message": msg,
            }
            _alerts.append(event)
            # Keep only last 200 alerts
            if len(_alerts) > 200:
                _alerts[:] = _alerts[-200:]


def push_sensing_snapshot(snapshot: Dict[str, Any]) -> None:
    """Called by the sensing tick loop to push a snapshot into the history
    buffer and evaluate alert rules.  This function is importable by the
    WebSocket server so it can feed the REST API."""
    _history.append(snapshot)
    _check_alert_rules(snapshot)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/current", response_model=SensingCurrentResponse)
async def get_current_sensing(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """현재 센싱 상태를 반환합니다.

    모바일 앱에서 폴링하여 실시간 감지 상태를 확인할 수 있습니다.
    """
    latest = _get_latest_sensing(request)
    if latest is None:
        raise HTTPException(status_code=503, detail="센싱 데이터가 아직 없습니다. 서버가 준비 중입니다.")

    features = latest.get("features", {})
    classification = latest.get("classification", {})

    return SensingCurrentResponse(
        timestamp=latest.get("timestamp", time.time()),
        source=latest.get("source", "unknown"),
        node_count=latest.get("node_count", 0),
        presence=classification.get("presence", False),
        motion_level=classification.get("motion_level", "absent"),
        confidence=classification.get("confidence", 0.0),
        features=SensingFeatures(
            mean_rssi=features.get("mean_rssi", 0),
            variance=features.get("variance", 0),
            std=features.get("std", 0),
            motion_band_power=features.get("motion_band_power", 0),
            breathing_band_power=features.get("breathing_band_power", 0),
            dominant_freq_hz=features.get("dominant_freq_hz", 0),
            spectral_power=features.get("spectral_power", 0),
            range=features.get("range", 0),
            iqr=features.get("iqr", 0),
            skewness=features.get("skewness", 0),
            kurtosis=features.get("kurtosis", 0),
        ),
        classification=ClassificationResult(
            motion_level=classification.get("motion_level", "absent"),
            presence=classification.get("presence", False),
            confidence=classification.get("confidence", 0.0),
        ),
    )


@router.get("/vitals", response_model=VitalsResponse)
async def get_vitals(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """바이탈 사인(호흡수, 심박수 추정)을 반환합니다.

    호흡 밴드 파워(0.1-0.5 Hz)에서 호흡수를, dominant frequency에서
    심박수를 추정합니다.  WiFi 센싱 기반이므로 참고용 수치입니다.
    """
    latest = _get_latest_sensing(request)
    if latest is None:
        raise HTTPException(status_code=503, detail="센싱 데이터가 아직 없습니다.")

    features = latest.get("features", {})
    breathing_power = features.get("breathing_band_power", 0.0)
    dominant_freq = features.get("dominant_freq_hz", 0.0)

    # Estimate breathing rate from dominant frequency if it's in the breathing band
    breathing_bpm: Optional[float] = None
    if 0.1 <= dominant_freq <= 0.5:
        breathing_bpm = round(dominant_freq * 60.0, 1)  # Hz -> BPM

    # Very rough heart rate heuristic from motion band harmonics
    heart_bpm: Optional[float] = None
    if dominant_freq > 0.8 and dominant_freq < 2.0:
        heart_bpm = round(dominant_freq * 60.0, 1)

    # Confidence based on breathing band strength
    confidence = min(1.0, breathing_power * 10.0)

    # Status classification
    if breathing_bpm is None:
        status = "unknown"
    elif breathing_bpm < 10:
        status = "low"
    elif breathing_bpm > 25:
        status = "elevated"
    else:
        status = "normal"

    return VitalsResponse(
        timestamp=latest.get("timestamp", time.time()),
        breathing_rate_bpm=breathing_bpm,
        heart_rate_bpm=heart_bpm,
        breathing_confidence=round(confidence, 3),
        status=status,
    )


@router.get("/nodes", response_model=List[NodeInfo])
async def get_nodes(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """연결된 ESP32 노드 목록을 반환합니다."""
    latest = _get_latest_sensing(request)
    if latest is None:
        return []

    nodes_raw = latest.get("nodes", [])
    return [
        NodeInfo(
            node_id=n.get("node_id", 0),
            rssi_dbm=n.get("rssi_dbm", 0),
            position=n.get("position", []),
            online=n.get("online", False),
            mean_amplitude=n.get("mean_amplitude", 0),
            subcarrier_count=n.get("subcarrier_count", 0),
            freq_mhz=n.get("freq_mhz", 0),
            source_addr=n.get("source_addr", ""),
            last_seen=n.get("last_seen", 0),
        )
        for n in nodes_raw
    ]


@router.get("/nodes/{node_id}", response_model=NodeInfo)
async def get_node(
    node_id: int,
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """특정 노드의 상세 정보를 반환합니다."""
    latest = _get_latest_sensing(request)
    if latest is None:
        raise HTTPException(status_code=503, detail="센싱 데이터가 아직 없습니다.")

    for n in latest.get("nodes", []):
        if n.get("node_id") == node_id:
            return NodeInfo(
                node_id=n.get("node_id", 0),
                rssi_dbm=n.get("rssi_dbm", 0),
                position=n.get("position", []),
                online=n.get("online", False),
                mean_amplitude=n.get("mean_amplitude", 0),
                subcarrier_count=n.get("subcarrier_count", 0),
                freq_mhz=n.get("freq_mhz", 0),
                source_addr=n.get("source_addr", ""),
                last_seen=n.get("last_seen", 0),
            )

    raise HTTPException(status_code=404, detail=f"노드 {node_id}를 찾을 수 없습니다.")


@router.get("/history")
async def get_sensing_history(
    minutes: int = Query(5, ge=1, le=60, description="조회할 기간 (분)"),
    limit: int = Query(100, ge=1, le=1000, description="최대 반환 개수"),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """최근 센싱 이력을 반환합니다.

    쿼리 파라미터로 기간(분)과 최대 개수를 지정할 수 있습니다.
    """
    cutoff = time.time() - (minutes * 60)
    filtered = [
        {
            "timestamp": s.get("timestamp"),
            "presence": s.get("classification", {}).get("presence", False),
            "motion_level": s.get("classification", {}).get("motion_level", "absent"),
            "confidence": s.get("classification", {}).get("confidence", 0),
            "mean_rssi": s.get("features", {}).get("mean_rssi", 0),
            "motion_band_power": s.get("features", {}).get("motion_band_power", 0),
            "node_count": s.get("node_count", 0),
        }
        for s in _history
        if s.get("timestamp", 0) >= cutoff
    ]

    # Return most recent entries (tail)
    result = filtered[-limit:]

    return {
        "period_minutes": minutes,
        "total_records": len(result),
        "data": result,
    }


@router.get("/alerts", response_model=List[AlertEvent])
async def get_alerts(
    limit: int = Query(50, ge=1, le=200, description="최대 알림 수"),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """현재 활성 알림 목록을 반환합니다."""
    return [AlertEvent(**a) for a in _alerts[-limit:]]


@router.post("/alerts", response_model=Dict[str, Any])
async def create_alert_rule(
    rule: AlertRule,
    current_user: Dict = Depends(require_auth),
):
    """새 알림 규칙을 등록합니다.

    조건 예시:
    - ``presence_detected``: 사람이 감지되면 알림
    - ``motion_active``: 움직임이 감지되면 알림
    - ``breathing_abnormal``: 호흡 이상 감지 시 알림
    """
    valid_conditions = {"presence_detected", "motion_active", "breathing_abnormal"}
    if rule.condition not in valid_conditions:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 조건입니다. 사용 가능: {valid_conditions}",
        )

    rule_dict = rule.model_dump()
    rule_dict["id"] = str(uuid.uuid4())
    rule_dict["created_at"] = time.time()
    _alert_rules.append(rule_dict)

    return {"message": "알림 규칙이 등록되었습니다.", "rule": rule_dict}


@router.get("/config", response_model=SensingConfigResponse)
async def get_sensing_config(
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """현재 센싱 설정을 반환합니다."""
    return SensingConfigResponse(**_sensing_config)


@router.put("/config", response_model=SensingConfigResponse)
async def update_sensing_config(
    update: SensingConfigUpdate,
    current_user: Dict = Depends(require_auth),
):
    """센싱 설정을 변경합니다 (인증 필요).

    변경 가능한 항목:
    - ``presence_variance_threshold``: 존재 감지 분산 임계값
    - ``motion_energy_threshold``: 모션 에너지 임계값
    - ``tick_interval_seconds``: 업데이트 주기 (초)
    - ``window_seconds``: 분석 윈도우 길이 (초)
    - ``source``: 데이터 소스 (auto/esp32/simulated)
    """
    update_data = update.model_dump(exclude_none=True)
    _sensing_config.update(update_data)
    logger.info("Sensing config updated: %s", update_data)

    return SensingConfigResponse(**_sensing_config)
