// RuView v0.5.1 — Premium Unified Dashboard Orchestrator
import { sensingService } from './services/sensing.service.js';
import { modelService } from './services/model.service.js';
import { DashboardRenderer } from './components/DashboardRenderer.js';

class WiFiDensePoseApp {
    constructor() {
        this.renderer = null;
        this.activeView = 'dashboard';
        this.isModelLoading = false;
    }

    async init() {
        console.log('🚀 Initializing RuView Premium Dashboard...');
        
        // Initialize Renderer
        this.renderer = new DashboardRenderer();
        
        // Initialize Services
        this.setupSensing();
        this.setupModelSelector();
        this.setupNavigation();
        
        // Start Sensing
        sensingService.start();
        
        console.log('✅ App initialized');
    }

    setupSensing() {
        sensingService.onData((data) => {
            this.updateDashboard(data);
        });

        sensingService.onStateChange((state) => {
            const statusPill = document.querySelector('.server-status-pill');
            if (statusPill) {
                statusPill.className = `server-status-pill ${state === 'connected' ? 'healthy' : 'warning'}`;
                statusPill.querySelector('.text').textContent = 
                    state === 'connected' ? 'Server Online' : 
                    state === 'reconnecting' ? 'Reconnecting...' : 'Server Offline';
            }
        });
    }

    updateDashboard(data) {
        // 1. Update Pose Skeleton
        if (data.pose_keypoints) {
            const confidence = data.classification?.confidence || 0;
            this.renderer.drawSkeleton(data.pose_keypoints, confidence);
            document.getElementById('confidence-display').textContent = `Conf: ${Math.round(confidence * 100)}%`;
        } else if (data.persons && data.persons.length > 0) {
            // Support multi-person if available
            const person = data.persons[0];
            const kps = person.keypoints.map(kp => [kp.x, kp.y, kp.z, kp.confidence]);
            this.renderer.drawSkeleton(kps, person.confidence);
            document.getElementById('confidence-display').textContent = `Conf: ${Math.round(person.confidence * 100)}%`;
        }

        // 2. Update FPS / Tick Info
        const fpsDisplay = document.getElementById('fps-display');
        if (fpsDisplay) {
            // Calculate FPS based on timestamps if needed, or just show source
            fpsDisplay.textContent = `${data.source.toUpperCase()}`;
        }

        // 3. Update Node Grid
        if (data.nodes) {
            let onlineCount = 0;
            data.nodes.forEach(node => {
                const card = document.querySelector(`.mini-node-card[data-node-id="${node.node_id}"]`);
                if (card) {
                    card.classList.remove('offline');
                    card.classList.add('online');
                    card.querySelector('.rssi-val').textContent = `${Math.round(node.rssi_dbm)} dBm`;
                    
                    // Update mini sparkline with mean amplitude
                    const meanAmp = node.amplitude.reduce((a, b) => a + b, 0) / (node.amplitude.length || 1);
                    this.renderer.updateNodeSparkline(node.node_id, meanAmp);
                    onlineCount++;
                }
            });
            document.getElementById('online-count').textContent = onlineCount;
        }

        // 4. Update Vitals
        if (data.vital_signs) {
            const vs = data.vital_signs;
            document.getElementById('heart-rate').textContent = Math.round(vs.heart_rate_bpm || 0);
            document.getElementById('resp-rate').innerHTML = `${Math.round(vs.breathing_rate_bpm || 0)} <small>rpm</small>`;
            
            // Pulse orb effect
            const orbPulse = document.getElementById('orb-pulse');
            if (orbPulse) {
                const offset = 283 - (283 * (vs.heart_rate_bpm / 120)); // Normalized to 120 BPM max
                orbPulse.style.strokeDashoffset = offset;
            }
            
            // Update vitals wave if we have raw spectral data or just mock a sine wave for visual feedback
            const waveVal = vs.hr_confidence * Math.sin(Date.now() / 100);
            this.renderer.updateVitalsWave(waveVal);
        }

        // 5. Update System Resources (Mocked or from data.model_status)
        if (data.model_status) {
            const cpu = data.model_status.cpu_usage || 0;
            const mem = data.model_status.memory_usage || 0;
            document.getElementById('cpu-fill').style.width = `${cpu}%`;
            document.getElementById('cpu-val').textContent = `${Math.round(cpu)}%`;
            document.getElementById('mem-fill').style.width = `${mem}%`;
            document.getElementById('mem-val').textContent = `${Math.round(mem)}%`;
        }
    }

    async setupModelSelector() {
        const select = document.getElementById('model-select');
        const loadBtn = document.getElementById('load-model-btn');
        
        try {
            const { models } = await modelService.listModels();
            select.innerHTML = models.map(m => `<option value="${m.id}">${m.name || m.id}</option>`).join('');
            
            const active = await modelService.getActiveModel();
            if (active) select.value = active.model_id;
        } catch (e) {
            console.error('Failed to load models:', e);
            select.innerHTML = '<option value="">No models found</option>';
        }

        loadBtn.addEventListener('click', async () => {
            if (this.isModelLoading) return;
            this.isModelLoading = true;
            loadBtn.textContent = '로드 중...';
            
            try {
                await modelService.loadModel(select.value);
                this.showToast('모델로드 성공', 'success');
            } catch (e) {
                this.showToast('모델로드 실패', 'error');
            } finally {
                this.isModelLoading = false;
                loadBtn.textContent = '적용';
            }
        });
    }

    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const view = item.dataset.view;
                this.switchView(view);
                
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');
            });
        });
    }

    switchView(viewId) {
        this.activeView = viewId;
        const views = document.querySelectorAll('.content-view');
        views.forEach(v => v.classList.remove('active'));
        
        const target = document.getElementById(`view-${viewId}`);
        if (target) target.classList.add('active');
        
        const titles = {
            dashboard: '시스템 통합 대시보드',
            mesh: '메쉬 네트워크 맵',
            analytics: '고급 신호 분석',
            settings: '시스템 초기 설정'
        };
        document.getElementById('view-title').textContent = titles[viewId] || 'RuView';
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const app = new WiFiDensePoseApp();
    app.init();
});

// Export for testing
export { WiFiDensePoseApp };