// WiFi DensePose Application - Main Entry Point (Simplified)

import { TabManager } from './components/TabManager.js';
import { DashboardTab } from './components/DashboardTab.js';
import { apiService } from './services/api.service.js';
import { wsService } from './services/websocket.service.js';
import { healthService } from './services/health.service.js';
import { sensingService } from './services/sensing.service.js';
import { backendDetector } from './utils/backend-detector.js';

class WiFiDensePoseApp {
  constructor() {
    this.components = {};
    this.isInitialized = false;
  }

  // Initialize application
  async init() {
    try {
      console.log('Initializing RuView by bin9208...');
      
      // Set up error handling
      this.setupErrorHandling();
      
      // Initialize services
      await this.initializeServices();
      
      // Initialize UI components
      this.initializeComponents();
      
      // Set up global event listeners
      this.setupEventListeners();
      
      this.isInitialized = true;
      console.log('RuView initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize application:', error);
      this.showGlobalError('Failed to initialize application. Please refresh the page.');
    }
  }

  // Initialize services
  async initializeServices() {
    // Add request interceptor for error handling
    apiService.addResponseInterceptor(async (response, url) => {
      if (!response.ok && response.status === 401) {
        console.warn('Authentication required for:', url);
      }
      return response;
    });

    // Detect backend availability and initialize accordingly
    const useMock = await backendDetector.shouldUseMockServer();
    
    if (useMock) {
      console.log('🧪 Initializing with mock server for testing');
      const { mockServer } = await import('./utils/mock-server.js');
      mockServer.start();
      this.showBackendStatus('Mock 서버 활성 — 테스트 모드', 'warning');
    } else {
      console.log('🔌 Connecting to backend...');

      try {
        const health = await healthService.checkLiveness();
        console.log('✅ Backend responding:', health);
        this.showBackendStatus('센싱 서버 연결됨', 'success');
      } catch (error) {
        console.warn('⚠️ Backend not available:', error.message);
        this.showBackendStatus('백엔드 미사용 — 센싱 서버를 시작하세요', 'warning');
      }

      // Start the sensing WebSocket service
      sensingService.start();
    }
  }

  // Initialize UI components
  initializeComponents() {
    const container = document.querySelector('.container');
    if (!container) {
      throw new Error('Main container not found');
    }

    // Initialize tab manager
    this.components.tabManager = new TabManager(container);
    this.components.tabManager.init();

    // Initialize tab components
    this.initializeTabComponents();

    // Set up tab change handling
    this.components.tabManager.onTabChange((newTab, oldTab) => {
      this.handleTabChange(newTab, oldTab);
    });
  }

  // Initialize individual tab components
  initializeTabComponents() {
    // Dashboard tab (Node Status)
    const dashboardContainer = document.getElementById('dashboard');
    if (dashboardContainer) {
      this.components.dashboard = new DashboardTab(dashboardContainer);
      this.components.dashboard.init().catch(error => {
        console.error('Failed to initialize dashboard:', error);
      });
    }

    // Training tab - lazy load to avoid breaking other tabs if import fails
    this.initTrainingTab();
  }

  // Lazy-load Training tab panels
  async initTrainingTab() {
    try {
      const [{ default: TrainingPanel }, { default: ModelPanel }] = await Promise.all([
        import('./components/TrainingPanel.js'),
        import('./components/ModelPanel.js')
      ]);

      const trainingContainer = document.getElementById('training-panel-container');
      if (trainingContainer) {
        this.components.trainingPanel = new TrainingPanel(trainingContainer);
      }

      const modelContainer = document.getElementById('model-panel-container');
      if (modelContainer) {
        this.components.modelPanel = new ModelPanel(modelContainer);
      }
    } catch (error) {
      console.error('Failed to load Training tab components:', error);
    }
  }

  // Handle tab changes
  handleTabChange(newTab, oldTab) {
    console.log(`Tab changed from ${oldTab} to ${newTab}`);
    
    switch (newTab) {
      case 'dashboard':
        // Dashboard auto-updates when visible
        break;

      case 'training':
        // Refresh panels when training tab becomes visible
        if (this.components.trainingPanel && typeof this.components.trainingPanel.refresh === 'function') {
          this.components.trainingPanel.refresh();
        }
        if (this.components.modelPanel && typeof this.components.modelPanel.refresh === 'function') {
          this.components.modelPanel.refresh();
        }
        break;
    }
  }

  // Set up global event listeners
  setupEventListeners() {
    window.addEventListener('resize', () => {
      this.handleResize();
    });

    document.addEventListener('visibilitychange', () => {
      this.handleVisibilityChange();
    });

    window.addEventListener('beforeunload', () => {
      this.cleanup();
    });
  }

  // Handle window resize
  handleResize() {
    const canvases = document.querySelectorAll('canvas');
    canvases.forEach(canvas => {
      const rect = canvas.parentElement.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    });
  }

  // Handle visibility change
  handleVisibilityChange() {
    if (document.hidden) {
      console.log('Page hidden, pausing updates');
      healthService.stopHealthMonitoring();
    } else {
      console.log('Page visible, resuming updates');
      healthService.startHealthMonitoring();
    }
  }

  // Set up error handling
  setupErrorHandling() {
    window.addEventListener('error', (event) => {
      if (event.error) {
        console.error('Global error:', event.error);
        this.showGlobalError('An unexpected error occurred');
      }
    });

    window.addEventListener('unhandledrejection', (event) => {
      if (event.reason) {
        console.error('Unhandled promise rejection:', event.reason);
        this.showGlobalError('An unexpected error occurred');
      }
    });
  }

  // Show backend status notification
  showBackendStatus(message, type) {
    let statusToast = document.getElementById('backendStatusToast');
    if (!statusToast) {
      statusToast = document.createElement('div');
      statusToast.id = 'backendStatusToast';
      statusToast.className = 'backend-status-toast';
      document.body.appendChild(statusToast);
    }

    statusToast.textContent = message;
    statusToast.className = `backend-status-toast ${type}`;
    statusToast.classList.add('show');

    const timeout = type === 'success' ? 3000 : 8000;
    setTimeout(() => {
      statusToast.classList.remove('show');
    }, timeout);
  }

  // Show global error message
  showGlobalError(message) {
    let errorToast = document.getElementById('globalErrorToast');
    if (!errorToast) {
      errorToast = document.createElement('div');
      errorToast.id = 'globalErrorToast';
      errorToast.className = 'error-toast';
      document.body.appendChild(errorToast);
    }

    errorToast.textContent = message;
    errorToast.classList.add('show');

    setTimeout(() => {
      errorToast.classList.remove('show');
    }, 5000);
  }

  // Clean up resources
  cleanup() {
    console.log('Cleaning up application resources...');
    
    Object.values(this.components).forEach(component => {
      if (component && typeof component.dispose === 'function') {
        component.dispose();
      }
    });

    wsService.disconnectAll();
    healthService.dispose();
  }

  // Public API
  getComponent(name) {
    return this.components[name];
  }

  isReady() {
    return this.isInitialized;
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.wifiDensePoseApp = new WiFiDensePoseApp();
  window.wifiDensePoseApp.init();
});

// Export for testing
export { WiFiDensePoseApp };