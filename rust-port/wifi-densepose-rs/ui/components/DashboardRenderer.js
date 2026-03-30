/**
 * DashboardRenderer.js
 * Handles high-performance rendering for the unified dashboard.
 */

export class DashboardRenderer {
    constructor() {
        this.skeletonCanvas = document.getElementById('skeleton-canvas');
        this.skeletonCtx = this.skeletonCanvas?.getContext('2d');
        this.vitalsCanvas = document.getElementById('vitals-wave-canvas');
        this.vitalsCtx = this.vitalsCanvas?.getContext('2d');
        
        this.nodeSparklines = new Map(); // nodeId -> { canvas, ctx, data }
        
        this._setupCanvases();
        this._initNodeSparklines();
        
        // Vitals buffer
        this.vitalsData = [];
        this.maxVitalsPoints = 100;
    }

    _setupCanvases() {
        if (this.skeletonCanvas) {
            this._resizeCanvas(this.skeletonCanvas);
            window.addEventListener('resize', () => this._resizeCanvas(this.skeletonCanvas));
        }
        if (this.vitalsCanvas) {
            this._resizeCanvas(this.vitalsCanvas);
            window.addEventListener('resize', () => this._resizeCanvas(this.vitalsCanvas));
        }
    }

    _resizeCanvas(canvas) {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
    }

    _initNodeSparklines() {
        const nodeCards = document.querySelectorAll('.mini-node-card');
        nodeCards.forEach(card => {
            const nodeId = parseInt(card.dataset.nodeId);
            const canvas = card.querySelector('.mini-sparkline');
            if (canvas) {
                const ctx = canvas.getContext('2d');
                this.nodeSparklines.set(nodeId, {
                    canvas,
                    ctx,
                    data: new Array(50).fill(0)
                });
            }
        });
    }

    /**
     * Render a skeleton on the main pose canvas
     * @param {Array} keypoints - 2D or 3D keypoints
     * @param {number} confidence - 0.0 to 1.0
     */
    drawSkeleton(keypoints, confidence) {
        if (!this.skeletonCtx) return;
        const ctx = this.skeletonCtx;
        const canvas = this.skeletonCanvas;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!keypoints || keypoints.length === 0) return;

        const scaleX = canvas.width;
        const scaleY = canvas.height;

        // Skeleton Connections (COCO format usually)
        const connections = [
            [0, 1], [0, 2], [1, 3], [2, 4], // Head
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
            [5, 11], [6, 12], [11, 12], // Torso
            [11, 13], [13, 15], [12, 14], [14, 16] // Legs
        ];

        ctx.lineWidth = 4 * window.devicePixelRatio;
        ctx.lineCap = 'round';
        ctx.strokeStyle = `rgba(0, 212, 255, ${0.3 + confidence * 0.7})`;
        ctx.shadowBlur = 10;
        ctx.shadowColor = 'rgba(0, 212, 255, 0.5)';

        connections.forEach(([i, j]) => {
            const p1 = keypoints[i];
            const p2 = keypoints[j];
            if (p1 && p2) {
                ctx.beginPath();
                ctx.moveTo(p1[0] * scaleX, p1[1] * scaleY);
                ctx.lineTo(p2[0] * scaleX, p2[1] * scaleY);
                ctx.stroke();
            }
        });

        // Joints
        ctx.fillStyle = '#fff';
        ctx.shadowBlur = 5;
        keypoints.forEach(kp => {
            ctx.beginPath();
            ctx.arc(kp[0] * scaleX, kp[1] * scaleY, 4 * window.devicePixelRatio, 0, Math.PI * 2);
            ctx.fill();
        });
        
        ctx.shadowBlur = 0;
    }

    /**
     * Update sparkline for a specific node
     */
    updateNodeSparkline(nodeId, value) {
        const spark = this.nodeSparklines.get(nodeId);
        if (!spark) return;

        spark.data.push(value);
        if (spark.data.length > 50) spark.data.shift();

        const { ctx, canvas, data } = spark;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.beginPath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#00d4ff';
        
        const step = canvas.width / (data.length - 1);
        const min = Math.min(...data);
        const max = Math.max(...data);
        const range = max - min || 1;

        data.forEach((v, i) => {
            const x = i * step;
            const y = canvas.height - ((v - min) / range) * canvas.height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    /**
     * Update vitals wave
     */
    updateVitalsWave(value) {
        if (!this.vitalsCtx) return;
        this.vitalsData.push(value);
        if (this.vitalsData.length > this.maxVitalsPoints) this.vitalsData.shift();

        const ctx = this.vitalsCtx;
        const canvas = this.vitalsCanvas;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#ff3e6d';
        ctx.shadowBlur = 10;
        ctx.shadowColor = 'rgba(255, 62, 109, 0.5)';

        const step = canvas.width / (this.vitalsData.length - 1);
        const min = Math.min(...this.vitalsData);
        const max = Math.max(...this.vitalsData);
        const range = max - min || 1;

        this.vitalsData.forEach((v, i) => {
            const x = i * step;
            const y = canvas.height - ((v - min) / range) * canvas.height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.shadowBlur = 0;
    }
}
