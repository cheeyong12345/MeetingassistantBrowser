/**
 * Audio Capture Module
 * Handles browser microphone capture using Web Audio API
 * Captures at 16kHz mono for Whisper compatibility
 */

class AudioCapture {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.sourceNode = null;
        this.processorNode = null;
        this.isRecording = false;
        this.onAudioData = null;
        this.onAudioLevel = null;
        this.onError = null;

        // Audio settings for Whisper compatibility
        this.targetSampleRate = 16000;
        this.bufferSize = 4096;
    }

    /**
     * Initialize audio capture and request microphone permission
     */
    async initialize() {
        try {
            // Request microphone permission with fallback support
            this.mediaStream = await this.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: this.targetSampleRate,
                    channelCount: 1
                }
            });

            // Create audio context with target sample rate
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.targetSampleRate
            });

            console.log(`Audio context created with sample rate: ${this.audioContext.sampleRate}Hz`);

            return true;

        } catch (error) {
            console.error('Failed to initialize audio:', error);
            if (this.onError) {
                this.onError('microphone_permission_denied', error.message);
            }
            return false;
        }
    }

    /**
     * Get user media with fallback for older browsers
     */
    async getUserMedia(constraints) {
        // Modern API
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            return await navigator.mediaDevices.getUserMedia(constraints);
        }

        // Legacy API fallback
        const legacyGetUserMedia = navigator.getUserMedia ||
                                    navigator.webkitGetUserMedia ||
                                    navigator.mozGetUserMedia ||
                                    navigator.msGetUserMedia;

        if (legacyGetUserMedia) {
            return new Promise((resolve, reject) => {
                legacyGetUserMedia.call(navigator, constraints, resolve, reject);
            });
        }

        throw new Error('getUserMedia is not supported in this browser');
    }

    /**
     * Start recording audio
     */
    async startRecording() {
        if (this.isRecording) {
            console.warn('Already recording');
            return false;
        }

        if (!this.mediaStream || !this.audioContext) {
            const initialized = await this.initialize();
            if (!initialized) {
                return false;
            }
        }

        try {
            // Create media stream source
            this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Create script processor for audio processing
            // Note: ScriptProcessorNode is deprecated but AudioWorklet requires separate file
            this.processorNode = this.audioContext.createScriptProcessor(
                this.bufferSize,
                1, // mono input
                1  // mono output
            );

            // Process audio chunks
            this.processorNode.onaudioprocess = (e) => {
                if (!this.isRecording) return;

                const inputData = e.inputBuffer.getChannelData(0);

                // Calculate audio level for visualization
                const level = this.calculateAudioLevel(inputData);
                if (this.onAudioLevel) {
                    this.onAudioLevel(level);
                }

                // Convert Float32Array to Int16Array (PCM)
                const pcmData = this.floatTo16BitPCM(inputData);

                // Send to callback
                if (this.onAudioData) {
                    this.onAudioData(pcmData);
                }
            };

            // Connect nodes
            this.sourceNode.connect(this.processorNode);
            this.processorNode.connect(this.audioContext.destination);

            // Resume audio context if suspended
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            this.isRecording = true;
            console.log('Recording started');
            return true;

        } catch (error) {
            console.error('Failed to start recording:', error);
            if (this.onError) {
                this.onError('recording_start_failed', error.message);
            }
            return false;
        }
    }

    /**
     * Stop recording audio
     */
    stopRecording() {
        if (!this.isRecording) {
            console.warn('Not currently recording');
            return;
        }

        this.isRecording = false;

        // Disconnect nodes
        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }

        if (this.processorNode) {
            this.processorNode.disconnect();
            this.processorNode = null;
        }

        console.log('Recording stopped');
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopRecording();

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }

        console.log('Audio capture cleaned up');
    }

    /**
     * Convert Float32Array to Int16Array (PCM 16-bit)
     */
    floatTo16BitPCM(float32Array) {
        const int16Array = new Int16Array(float32Array.length);

        for (let i = 0; i < float32Array.length; i++) {
            // Clamp to [-1, 1] range
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            // Convert to 16-bit integer
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        return int16Array;
    }

    /**
     * Calculate audio level (RMS)
     */
    calculateAudioLevel(audioData) {
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        return Math.sqrt(sum / audioData.length);
    }

    /**
     * Check if browser supports required APIs
     */
    static isSupported() {
        // Check for AudioContext support (all modern browsers)
        const hasAudioContext = !!(window.AudioContext || window.webkitAudioContext);

        // Check for getUserMedia support
        // Note: navigator.mediaDevices requires secure context (HTTPS or localhost)
        const hasMediaDevices = !!(
            navigator.mediaDevices &&
            navigator.mediaDevices.getUserMedia
        );

        // Fallback: Check for legacy getUserMedia (should work in all contexts)
        const hasLegacyGetUserMedia = !!(
            navigator.getUserMedia ||
            navigator.webkitGetUserMedia ||
            navigator.mozGetUserMedia ||
            navigator.msGetUserMedia
        );

        return hasAudioContext && (hasMediaDevices || hasLegacyGetUserMedia);
    }

    /**
     * Check if running in secure context (HTTPS or localhost)
     */
    static isSecureContext() {
        return window.isSecureContext ||
               location.protocol === 'https:' ||
               location.hostname === 'localhost' ||
               location.hostname === '127.0.0.1' ||
               location.hostname === '[::1]';
    }

    /**
     * Get detailed support information for debugging
     */
    static getSupportInfo() {
        return {
            isSecureContext: AudioCapture.isSecureContext(),
            hasMediaDevices: !!(navigator.mediaDevices),
            hasGetUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            hasAudioContext: !!(window.AudioContext || window.webkitAudioContext),
            protocol: location.protocol,
            hostname: location.hostname
        };
    }

    /**
     * Get audio context sample rate
     */
    getSampleRate() {
        return this.audioContext ? this.audioContext.sampleRate : this.targetSampleRate;
    }

    /**
     * Get recording status
     */
    getStatus() {
        return {
            isRecording: this.isRecording,
            sampleRate: this.getSampleRate(),
            hasPermission: !!this.mediaStream,
            contextState: this.audioContext ? this.audioContext.state : 'closed'
        };
    }
}

// Export for use in other modules
window.AudioCapture = AudioCapture;
