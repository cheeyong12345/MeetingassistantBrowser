/**
 * Main Application Logic
 * Orchestrates audio capture, WebSocket communication, and UI updates
 */

class MeetingAssistantApp {
    constructor() {
        this.audioCapture = null;
        this.wsClient = null;
        this.sessionId = null;
        this.isRecording = false;
        this.transcriptSegments = [];

        // UI elements
        this.elements = {};

        // Initialize
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        console.log('Initializing Meeting Assistant App');

        // Check browser compatibility
        if (!AudioCapture.isSupported()) {
            this.showError('Your browser does not support required audio features. Please use Chrome, Firefox, or Edge.');
            return;
        }

        // Cache UI elements
        this.cacheElements();

        // Setup event listeners
        this.setupEventListeners();

        // Load system status
        this.loadSystemStatus();

        console.log('App initialized successfully');
    }

    /**
     * Cache UI elements
     */
    cacheElements() {
        this.elements = {
            // Meeting controls
            meetingTitleInput: document.getElementById('meeting-title'),
            participantsInput: document.getElementById('participants'),
            startBtn: document.getElementById('start-meeting-btn'),
            stopBtn: document.getElementById('stop-meeting-btn'),

            // Recording status
            recordingStatus: document.getElementById('recording-status'),
            connectionStatus: document.getElementById('connection-status'),
            audioLevelMeter: document.getElementById('audio-level-meter'),
            audioLevelBar: document.getElementById('audio-level-bar'),

            // Transcript
            transcriptContainer: document.getElementById('transcript-container'),
            transcriptContent: document.getElementById('transcript-content'),

            // Summary
            summaryContainer: document.getElementById('summary-container'),
            summaryContent: document.getElementById('summary-content'),

            // Status
            systemStatus: document.getElementById('system-status'),
            errorContainer: document.getElementById('error-container'),
            errorMessage: document.getElementById('error-message')
        };
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Meeting controls
        if (this.elements.startBtn) {
            this.elements.startBtn.addEventListener('click', () => this.startMeeting());
        }

        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener('click', () => this.stopMeeting());
        }

        // Handle page unload
        window.addEventListener('beforeunload', (e) => {
            if (this.isRecording) {
                e.preventDefault();
                e.returnValue = 'Meeting is in progress. Are you sure you want to leave?';
            }
        });

        // Cleanup on page unload
        window.addEventListener('unload', () => {
            this.cleanup();
        });
    }

    /**
     * Load system status from API
     */
    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();

            this.updateSystemStatus(status);

        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }

    /**
     * Update system status display
     */
    updateSystemStatus(status) {
        if (!this.elements.systemStatus) return;

        const sttStatus = status.stt?.initialized ? 'Ready' : 'Not Available';
        const summaryStatus = status.summarization?.initialized ? 'Ready' : 'Not Available';

        this.elements.systemStatus.innerHTML = `
            <div class="status-item">
                <span class="status-label">STT Engine:</span>
                <span class="status-value ${status.stt?.initialized ? 'ready' : 'error'}">${sttStatus}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Summarization:</span>
                <span class="status-value ${status.summarization?.initialized ? 'ready' : 'error'}">${summaryStatus}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Audio Mode:</span>
                <span class="status-value ready">Browser Microphone</span>
            </div>
        `;
    }

    /**
     * Start meeting
     */
    async startMeeting() {
        try {
            const title = this.elements.meetingTitleInput?.value || '';
            const participants = this.elements.participantsInput?.value || '';

            // Create meeting on server
            const formData = new FormData();
            formData.append('title', title);
            formData.append('participants', participants);

            const response = await fetch('/api/meeting/start', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!result.success) {
                this.showError(`Failed to start meeting: ${result.error}`);
                return;
            }

            this.sessionId = result.session_id;
            console.log('Meeting started:', this.sessionId);

            // Initialize audio capture
            this.audioCapture = new AudioCapture();
            const audioInitialized = await this.audioCapture.initialize();

            if (!audioInitialized) {
                this.showError('Failed to access microphone. Please grant permission and try again.');
                return;
            }

            // Setup audio callbacks
            this.audioCapture.onAudioData = (audioData) => {
                if (this.wsClient && this.wsClient.isConnected) {
                    this.wsClient.sendAudio(audioData);
                }
            };

            this.audioCapture.onAudioLevel = (level) => {
                this.updateAudioLevel(level);
            };

            this.audioCapture.onError = (type, message) => {
                this.showError(`Audio error: ${message}`);
            };

            // Connect WebSocket
            this.wsClient = new WebSocketClient(this.sessionId);

            this.wsClient.onConnected = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus('Connected', 'connected');

                // Start audio recording
                this.audioCapture.startRecording();
                this.isRecording = true;
                this.updateRecordingStatus('Recording', 'recording');

                // Update UI
                this.updateMeetingUI(true);
            };

            this.wsClient.onDisconnected = (code, reason) => {
                console.log('WebSocket disconnected:', code, reason);
                this.updateConnectionStatus('Disconnected', 'disconnected');

                if (this.isRecording) {
                    this.showError('Connection lost. Attempting to reconnect...');
                }
            };

            this.wsClient.onTranscription = (data) => {
                this.addTranscription(data.text, data.timestamp, data.confidence);
            };

            this.wsClient.onError = (type, message) => {
                this.showError(`WebSocket error: ${message}`);
            };

            // Connect to WebSocket
            this.wsClient.connect();

        } catch (error) {
            console.error('Error starting meeting:', error);
            this.showError(`Failed to start meeting: ${error.message}`);
        }
    }

    /**
     * Stop meeting
     */
    async stopMeeting() {
        try {
            // Stop audio recording
            if (this.audioCapture) {
                this.audioCapture.stopRecording();
            }

            // Disconnect WebSocket
            if (this.wsClient) {
                this.wsClient.disconnect();
            }

            this.isRecording = false;
            this.updateRecordingStatus('Stopped', 'stopped');
            this.updateConnectionStatus('Disconnected', 'disconnected');

            // Stop meeting on server
            if (this.sessionId) {
                const formData = new FormData();
                formData.append('session_id', this.sessionId);

                const response = await fetch('/api/meeting/stop', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    console.log('Meeting stopped successfully');

                    // Display summary if available
                    if (result.meeting?.summary) {
                        this.displaySummary(result.meeting.summary);
                    }
                } else {
                    this.showError(`Failed to stop meeting: ${result.error}`);
                }
            }

            // Cleanup
            this.cleanup();

            // Update UI
            this.updateMeetingUI(false);

        } catch (error) {
            console.error('Error stopping meeting:', error);
            this.showError(`Failed to stop meeting: ${error.message}`);
        }
    }

    /**
     * Add transcription to display
     */
    addTranscription(text, timestamp, confidence) {
        if (!text) return;

        const segment = {
            text: text,
            timestamp: timestamp || Date.now() / 1000,
            confidence: confidence || 0
        };

        this.transcriptSegments.push(segment);

        // Update transcript display
        this.updateTranscriptDisplay();
    }

    /**
     * Update transcript display
     */
    updateTranscriptDisplay() {
        if (!this.elements.transcriptContent) return;

        const transcriptHTML = this.transcriptSegments.map(segment => {
            const time = new Date(segment.timestamp * 1000).toLocaleTimeString();
            const confidenceClass = segment.confidence > 0.8 ? 'high' : segment.confidence > 0.5 ? 'medium' : 'low';

            return `
                <div class="transcript-segment ${confidenceClass}">
                    <span class="timestamp">${time}</span>
                    <span class="text">${this.escapeHtml(segment.text)}</span>
                </div>
            `;
        }).join('');

        this.elements.transcriptContent.innerHTML = transcriptHTML;

        // Show transcript container
        if (this.elements.transcriptContainer) {
            this.elements.transcriptContainer.style.display = 'block';
        }

        // Auto-scroll to bottom
        this.elements.transcriptContent.scrollTop = this.elements.transcriptContent.scrollHeight;
    }

    /**
     * Display summary
     */
    displaySummary(summary) {
        if (!this.elements.summaryContent) return;

        this.elements.summaryContent.textContent = summary;

        if (this.elements.summaryContainer) {
            this.elements.summaryContainer.style.display = 'block';
        }
    }

    /**
     * Update audio level meter
     */
    updateAudioLevel(level) {
        if (!this.elements.audioLevelBar) return;

        // Convert to percentage (0-100)
        const percentage = Math.min(100, level * 200);

        this.elements.audioLevelBar.style.width = `${percentage}%`;

        // Color coding
        if (percentage > 70) {
            this.elements.audioLevelBar.style.backgroundColor = '#ef4444';
        } else if (percentage > 40) {
            this.elements.audioLevelBar.style.backgroundColor = '#10b981';
        } else {
            this.elements.audioLevelBar.style.backgroundColor = '#6b7280';
        }
    }

    /**
     * Update recording status
     */
    updateRecordingStatus(status, className) {
        if (!this.elements.recordingStatus) return;

        this.elements.recordingStatus.textContent = status;
        this.elements.recordingStatus.className = `status ${className}`;
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(status, className) {
        if (!this.elements.connectionStatus) return;

        this.elements.connectionStatus.textContent = status;
        this.elements.connectionStatus.className = `status ${className}`;
    }

    /**
     * Update meeting UI state
     */
    updateMeetingUI(isActive) {
        if (this.elements.startBtn) {
            this.elements.startBtn.disabled = isActive;
        }

        if (this.elements.stopBtn) {
            this.elements.stopBtn.disabled = !isActive;
        }

        if (this.elements.meetingTitleInput) {
            this.elements.meetingTitleInput.disabled = isActive;
        }

        if (this.elements.participantsInput) {
            this.elements.participantsInput.disabled = isActive;
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        console.error(message);

        if (this.elements.errorMessage) {
            this.elements.errorMessage.textContent = message;
        }

        if (this.elements.errorContainer) {
            this.elements.errorContainer.style.display = 'block';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                this.elements.errorContainer.style.display = 'none';
            }, 5000);
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.audioCapture) {
            this.audioCapture.cleanup();
            this.audioCapture = null;
        }

        if (this.wsClient) {
            this.wsClient.disconnect();
            this.wsClient = null;
        }

        this.sessionId = null;
        this.isRecording = false;
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MeetingAssistantApp();
});
