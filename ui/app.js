// RuView — WiFi DensePose Dashboard
// Self-contained: WebSocket + Canvas Skeleton + REST API

(() => {
'use strict';

// ── COCO 17-keypoint skeleton ───────────────────────────────────
const KEYPOINT_NAMES = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
];

const SKELETON_EDGES = [
    [0,1],[0,2],[1,3],[2,4],               // head
    [5,6],                                   // shoulders
    [5,7],[7,9],[6,8],[8,10],               // arms
    [5,11],[6,12],[11,12],                  // torso
    [11,13],[13,15],[12,14],[14,16]         // legs
];

const EDGE_COLORS = {
    head:  '#00d4ff',
    arm_l: '#22d3a8',
    arm_r: '#a855f7',
    torso: '#f59e0b',
    leg_l: '#22d3a8',
    leg_r: '#a855f7'
};

function edgeColor(i, j) {
    if (i <= 4 || j <= 4) return EDGE_COLORS.head;
    if ([5,7,9].includes(i) || [5,7,9].includes(j)) return EDGE_COLORS.arm_l;
    if ([6,8,10].includes(i) || [6,8,10].includes(j)) return EDGE_COLORS.arm_r;
    if ((i===5&&j===11)||(i===6&&j===12)||(i===11&&j===12)||(i===5&&j===6)) return EDGE_COLORS.torso;
    if ([11,13,15].includes(i) || [11,13,15].includes(j)) return EDGE_COLORS.leg_l;
    if ([12,14,16].includes(i) || [12,14,16].includes(j)) return EDGE_COLORS.leg_r;
    return '#666';
}

// ── DOM refs ────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const canvas = $('pose-canvas');
const ctx = canvas.getContext('2d');

// ── State ───────────────────────────────────────────────────────
let ws = null;
let reconnectTimer = null;
let reconnectDelay = 1000;
const MAX_RECONNECT = 30000;
let lastData = null;
let nodeMap = {};          // node_id → latest node info
let animFrame = null;

// ── WebSocket ───────────────────────────────────────────────────

function wsUrl() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws/sensing`;
}

function connectWS() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

    setConnStatus('connecting', '연결 중...');

    try {
        ws = new WebSocket(wsUrl());
    } catch (e) {
        scheduleReconnect();
        return;
    }

    ws.onopen = () => {
        reconnectDelay = 1000;
        setConnStatus('online', '서버 연결됨');
        toast('서버 연결 성공', 'ok');
    };

    ws.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            if (data.type === 'pong') return;
            lastData = data;
            handleUpdate(data);
        } catch (_) {}
    };

    ws.onclose = () => {
        setConnStatus('offline', '연결 끊김');
        scheduleReconnect();
    };

    ws.onerror = () => {
        ws.close();
    };
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectWS();
    }, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT);
}

function setConnStatus(cls, text) {
    const pill = $('conn-status');
    pill.className = `conn-pill ${cls}`;
    pill.querySelector('.conn-label').textContent = text;
}

// ── Data Handler ────────────────────────────────────────────────

function handleUpdate(d) {
    // Source badge
    $('badge-source').textContent = (d.source || '--').toUpperCase();

    // Persons
    const persons = d.persons || [];
    $('badge-persons').textContent = `${d.estimated_persons || persons.length}명`;
    $('det-persons').textContent = d.estimated_persons || persons.length;

    // Pose — extract keypoints
    let kps = null;
    let conf = 0;
    if (persons.length > 0) {
        const p = persons[0];
        kps = p.keypoints.map(k => [k.x, k.y, k.confidence]);
        conf = p.confidence;
    } else if (d.pose_keypoints && d.pose_keypoints.length >= 17) {
        kps = d.pose_keypoints.map(k => [k[0], k[1], k[3] || k[2] || 0.5]);
        conf = d.classification ? d.classification.confidence : 0;
    }

    if (kps) {
        drawSkeleton(kps);
        $('badge-conf').textContent = `${Math.round(conf * 100)}%`;
    }

    // Vital Signs
    if (d.vital_signs) {
        const vs = d.vital_signs;
        const hr = vs.heart_rate_bpm || 0;
        const br = vs.breathing_rate_bpm || 0;
        const hrConf = vs.hr_confidence || vs.heart_confidence || 0;
        const brConf = vs.breathing_confidence || 0;

        $('hr-val').textContent = hr > 0 ? Math.round(hr) : '--';
        $('br-val').textContent = br > 0 ? Math.round(br) : '--';

        // Ring arcs (max HR=150, max BR=40)
        setRingArc('hr-arc', hr / 150);
        setRingArc('br-arc', br / 40);

        // Confidence bars
        $('hr-conf-bar').style.width = `${hrConf * 100}%`;
        $('hr-conf-text').textContent = `HR 신뢰도 ${Math.round(hrConf * 100)}%`;
        $('br-conf-bar').style.width = `${brConf * 100}%`;
        $('br-conf-text').textContent = `BR 신뢰도 ${Math.round(brConf * 100)}%`;
    }

    // Classification
    if (d.classification) {
        const c = d.classification;
        const motionEl = $('det-motion');
        motionEl.textContent = c.motion_level || '--';
        motionEl.className = 'detect-val ' + motionClass(c.motion_level);

        const presEl = $('det-presence');
        presEl.textContent = c.presence ? 'Yes' : 'No';
        presEl.className = 'detect-val ' + (c.presence ? 'presence-yes' : 'presence-no');

        $('det-conf-bar').style.width = `${(c.confidence || 0) * 100}%`;
    }

    // Posture
    $('det-posture').textContent = d.posture || '--';

    // Signal Quality
    if (d.signal_quality_score != null) {
        $('sig-quality-bar').style.width = `${d.signal_quality_score * 100}%`;
    }

    // Features
    if (d.features) {
        const f = d.features;
        $('sig-rssi').textContent = f.mean_rssi != null ? `${f.mean_rssi.toFixed(1)} dBm` : '-- dBm';
        $('sig-var').textContent = f.variance != null ? f.variance.toFixed(3) : '--';
        $('sig-motion').textContent = f.motion_band_power != null ? f.motion_band_power.toFixed(4) : '--';
        $('sig-breath').textContent = f.breathing_band_power != null ? f.breathing_band_power.toFixed(4) : '--';
        $('sig-freq').textContent = f.dominant_freq_hz != null ? `${f.dominant_freq_hz.toFixed(2)} Hz` : '-- Hz';
        $('sig-power').textContent = f.spectral_power != null ? f.spectral_power.toFixed(4) : '--';
    }

    // Nodes
    if (d.nodes) {
        d.nodes.forEach(n => { nodeMap[n.node_id] = n; });
        renderNodes();
    }
}

function motionClass(level) {
    if (!level) return '';
    if (level === 'active') return 'motion-active';
    if (level === 'present_still') return 'motion-still';
    return 'motion-absent';
}

function setRingArc(id, ratio) {
    const el = $(id);
    if (!el) return;
    const circumference = 213.6; // 2 * PI * 34
    const clamped = Math.max(0, Math.min(1, ratio));
    el.style.strokeDashoffset = circumference * (1 - clamped);
}

// ── Nodes Renderer ──────────────────────────────────────────────

function renderNodes() {
    const list = $('node-list');
    const ids = Object.keys(nodeMap).sort((a,b) => a - b);
    $('badge-nodes').textContent = `${ids.length} online`;

    if (ids.length === 0) {
        list.innerHTML = '<div class="empty-msg">노드 대기 중...</div>';
        return;
    }

    list.innerHTML = ids.map(id => {
        const n = nodeMap[id];
        const rssi = Math.round(n.rssi_dbm);
        const subs = n.subcarrier_count || (n.amplitude ? n.amplitude.length : 0);
        return `<div class="node-row active">
            <div class="node-dot"></div>
            <span class="node-name">Node ${n.node_id}</span>
            <span class="node-rssi">${rssi} dBm</span>
            <span class="node-subs">${subs}sc</span>
        </div>`;
    }).join('');
}

// ── Skeleton Canvas ─────────────────────────────────────────────

function resizeCanvas() {
    const wrap = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = wrap.clientWidth * dpr;
    canvas.height = wrap.clientHeight * dpr;
    ctx.scale(dpr, dpr);
}

function drawSkeleton(kps) {
    const w = canvas.parentElement.clientWidth;
    const h = canvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, w, h);

    // Background grid
    drawGrid(w, h);

    if (!kps || kps.length < 17) return;

    // Map normalized [0,1] coordinates to canvas (with padding)
    const pad = 40;
    const pw = w - pad * 2;
    const ph = h - pad * 2;
    const pts = kps.map(([x, y, c]) => ({
        x: pad + x * pw,
        y: pad + y * ph,
        c: c
    }));

    // Draw edges
    SKELETON_EDGES.forEach(([i, j]) => {
        const a = pts[i], b = pts[j];
        if (a.c < 0.1 || b.c < 0.1) return;

        const alpha = Math.min(a.c, b.c);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = edgeColor(i, j);
        ctx.globalAlpha = alpha * 0.9;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.stroke();
        ctx.globalAlpha = 1;
    });

    // Draw joints
    pts.forEach((p, i) => {
        if (p.c < 0.1) return;

        // Glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = edgeColor(i, i);
        ctx.globalAlpha = p.c * 0.2;
        ctx.fill();

        // Dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.globalAlpha = p.c * 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
    });
}

function drawGrid(w, h) {
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    const step = 40;
    for (let x = step; x < w; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = step; y < h; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }
}

// ── Model Selector ──────────────────────────────────────────────

async function loadModels() {
    try {
        const res = await fetch('/api/v1/models');
        if (!res.ok) return;
        const data = await res.json();
        const sel = $('model-select');
        const models = data.models || data || [];
        if (models.length === 0) {
            sel.innerHTML = '<option value="">모델 없음</option>';
            return;
        }
        sel.innerHTML = models.map(m =>
            `<option value="${m.id}">${m.name || m.id}</option>`
        ).join('');

        // Check active model
        try {
            const activeRes = await fetch('/api/v1/models/active');
            if (activeRes.ok) {
                const active = await activeRes.json();
                if (active && active.model_id) sel.value = active.model_id;
            }
        } catch (_) {}
    } catch (_) {
        $('model-select').innerHTML = '<option value="">서버 오프라인</option>';
    }
}

async function onLoadModel() {
    const btn = $('btn-load-model');
    const sel = $('model-select');
    const modelId = sel.value;
    if (!modelId) return;

    btn.disabled = true;
    btn.textContent = '로딩...';

    try {
        const res = await fetch('/api/v1/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });
        if (res.ok) {
            toast('모델 로드 완료', 'ok');
        } else {
            toast('모델 로드 실패', 'err');
        }
    } catch (e) {
        toast('서버 응답 없음', 'err');
    } finally {
        btn.disabled = false;
        btn.textContent = '로드';
    }
}

// ── Toast ────────────────────────────────────────────────────────

function toast(msg, type = 'ok') {
    const area = $('toast-area');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    area.appendChild(el);
    setTimeout(() => el.remove(), 3000);
}

// ── Recording ───────────────────────────────────────────────────

let recActive = false;
let recStartTime = null;
let recTimerInterval = null;

function apiBase() {
    return `${location.protocol}//${location.host}`;
}

async function toggleRecording() {
    const btn = $('btn-record');
    const timer = $('rec-timer');

    if (!recActive) {
        // Start recording
        try {
            const id = `rec_${Date.now()}`;
            const resp = await fetch(`${apiBase()}/api/v1/recording/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id }),
            });
            const data = await resp.json();
            if (data.success) {
                recActive = true;
                recStartTime = Date.now();
                btn.classList.add('recording');
                btn.textContent = 'STOP';
                timer.style.display = 'inline';
                recTimerInterval = setInterval(updateRecTimer, 1000);
                toast(`녹화 시작: ${data.recording_id}`, 'ok');
            } else {
                toast(`녹화 실패: ${data.error}`, 'err');
            }
        } catch (e) {
            toast('서버 응답 없음', 'err');
        }
    } else {
        // Stop recording
        try {
            const resp = await fetch(`${apiBase()}/api/v1/recording/stop`, {
                method: 'POST',
            });
            const data = await resp.json();
            recActive = false;
            btn.classList.remove('recording');
            btn.textContent = 'REC';
            timer.style.display = 'none';
            clearInterval(recTimerInterval);
            const frames = data.frames_recorded || '?';
            const dur = data.duration_secs || '?';
            toast(`녹화 완료: ${frames}프레임, ${dur}초`, 'ok');
        } catch (e) {
            toast('녹화 중지 실패', 'err');
        }
    }
}

function updateRecTimer() {
    if (!recStartTime) return;
    const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
    const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const ss = String(elapsed % 60).padStart(2, '0');
    $('rec-timer').textContent = `${mm}:${ss}`;
}

// ── Init ─────────────────────────────────────────────────────────

function init() {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    $('btn-load-model').addEventListener('click', onLoadModel);
    $('btn-record').addEventListener('click', toggleRecording);

    connectWS();
    loadModels();

    // Ping keepalive
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        }
    }, 30000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
