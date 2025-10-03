/**
 * WebSocket Client Module
 * Handles WebSocket communication with server for audio streaming
 */

class WebSocketClient {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.reconnectTimer = null;

        // Callbacks
        this.onConnected = null;
        this.onDisconnected = null;
        this.onTranscription = null;
        this.onAudioLevel = null;
        this.onError = null;
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        try {
            // Determine WebSocket URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/audio/${this.sessionId}`;

            console.log(`Connecting to WebSocket: ${wsUrl}`);

            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = this.handleOpen.bind(this);
            this.ws.onclose = this.handleClose.bind(this);
            this.ws.onerror = this.handleError.bind(this);
            this.ws.onmessage = this.handleMessage.bind(this);

        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            if (this.onError) {
                this.onError('connection_failed', error.message);
            }
        }
    }

    /**
     * Handle WebSocket open event
     */
    handleOpen(event) {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;

        if (this.onConnected) {
            this.onConnected();
        }
    }

    /**
     * Handle WebSocket close event
     */
    handleClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);
        this.isConnected = false;

        if (this.onDisconnected) {
            this.onDisconnected(event.code, event.reason);
        }

        // Attempt reconnection if not cleanly closed
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket error event
     */
    handleError(error) {
        console.error('WebSocket error:', error);

        if (this.onError) {
            this.onError('websocket_error', 'Connection error occurred');
        }
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(event) {
        try {
            // Parse JSON message
            const message = JSON.parse(event.data);

            switch (message.type) {
                case 'connected':
                    console.log('Server confirmed connection:', message.message);
                    break;

                case 'transcription':
                    if (this.onTranscription) {
                        this.onTranscription({
                            text: message.text,
                            timestamp: message.timestamp,
                            confidence: message.confidence
                        });
                    }
                    break;

                case 'audio_level':
                    if (this.onAudioLevel) {
                        this.onAudioLevel(message.level);
                    }
                    break;

                case 'error':
                    console.error('Server error:', message.message);
                    if (this.onError) {
                        this.onError('server_error', message.message);
                    }
                    break;

                case 'pong':
                    // Heartbeat response
                    break;

                case 'transcript':
                    // Full transcript received
                    console.log('Received full transcript:', message.text);
                    break;

                default:
                    console.warn('Unknown message type:', message.type);
            }

        } catch (error) {
            console.error('Failed to parse message:', error);
        }
    }

    /**
     * Send audio data to server
     */
    sendAudio(audioData) {
        if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('Cannot send audio: WebSocket not connected');
            return false;
        }

        try {
            // Send as binary data
            this.ws.send(audioData.buffer);
            return true;

        } catch (error) {
            console.error('Failed to send audio:', error);
            if (this.onError) {
                this.onError('send_failed', error.message);
            }
            return false;
        }
    }

    /**
     * Send JSON control message
     */
    sendMessage(message) {
        if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('Cannot send message: WebSocket not connected');
            return false;
        }

        try {
            this.ws.send(JSON.stringify(message));
            return true;

        } catch (error) {
            console.error('Failed to send message:', error);
            return false;
        }
    }

    /**
     * Request full transcript from server
     */
    requestTranscript() {
        return this.sendMessage({ type: 'get_transcript' });
    }

    /**
     * Send heartbeat ping
     */
    sendPing() {
        return this.sendMessage({ type: 'ping' });
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * this.reconnectAttempts;

        console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);

        this.reconnectTimer = setTimeout(() => {
            console.log('Attempting to reconnect...');
            this.connect();
        }, delay);
    }

    /**
     * Disconnect from WebSocket
     */
    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close(1000, 'Client requested disconnect');
            this.ws = null;
        }

        this.isConnected = false;
        this.reconnectAttempts = 0;
    }

    /**
     * Get connection status
     */
    getStatus() {
        return {
            isConnected: this.isConnected,
            readyState: this.ws ? this.ws.readyState : WebSocket.CLOSED,
            reconnectAttempts: this.reconnectAttempts
        };
    }

    /**
     * Get readable connection state
     */
    getReadableState() {
        if (!this.ws) return 'Not initialized';

        switch (this.ws.readyState) {
            case WebSocket.CONNECTING: return 'Connecting';
            case WebSocket.OPEN: return 'Connected';
            case WebSocket.CLOSING: return 'Closing';
            case WebSocket.CLOSED: return 'Closed';
            default: return 'Unknown';
        }
    }
}

// Export for use in other modules
window.WebSocketClient = WebSocketClient;
