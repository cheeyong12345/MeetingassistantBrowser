# Meeting Assistant - Browser Version

A browser-based meeting transcription and summarization tool that captures audio from your laptop's microphone using Web Audio API instead of server-side audio capture.

## Overview

This is the browser-based version of Meeting Assistant that differs from the server-based version in how audio is captured:

- **Server Version**: Captures audio from the server's microphone using PyAudio
- **Browser Version**: Captures audio from the browser (user's laptop microphone) using Web Audio API

Both versions use the same backend STT (Speech-to-Text) and summarization engines.

## Architecture

### Audio Flow

```
Browser Microphone
       ↓
Web Audio API (16kHz, Mono)
       ↓
JavaScript Audio Processing (Float32 → Int16 PCM)
       ↓
WebSocket (Binary)
       ↓
FastAPI Backend
       ↓
STT Engine (Whisper/WhisperCPP/Vosk)
       ↓
Real-time Transcription
       ↓
WebSocket (JSON)
       ↓
Browser UI Display
```

### Key Components

1. **Frontend (Browser)**
   - `audio-capture.js`: Web Audio API implementation for microphone capture
   - `websocket-client.js`: WebSocket communication with server
   - `app.js`: Main application logic and UI management
   - `style.css`: Modern, responsive UI styling

2. **Backend (Server)**
   - `web_app_browser.py`: FastAPI application with WebSocket endpoints
   - `src/stt/`: Speech-to-text engines (Whisper, WhisperCPP, Vosk)
   - `src/summarization/`: Summarization engines (Qwen, Ollama, OpenAI)
   - `src/config.py`: Configuration management

## Installation

### Prerequisites

- Python 3.10 or higher
- Modern web browser (Chrome 56+, Firefox 52+, Edge 79+, Safari 14.1+)
- No PyAudio required (audio is captured in browser)

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd /home/amd/MeetingassistantBrowser
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # OR
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download STT models** (if not already present)

   For Whisper.cpp:
   ```bash
   mkdir -p models
   # Download your preferred model size (tiny, base, small, medium, large)
   # See: https://github.com/ggerganov/whisper.cpp#models
   ```

   For Vosk:
   ```bash
   # Download from: https://alphacephei.com/vosk/models
   # Extract to models/vosk-model-en-us-0.22/
   ```

5. **Configure settings** (optional)

   Edit `config.yaml` to customize:
   - STT engine and model size
   - Summarization engine
   - Server host and port
   - Audio settings

## Running the Application

### Start the Server

```bash
python web_app_browser.py
```

The server will start at `http://localhost:8000` (default).

### Access the Web Interface

1. Open your browser and navigate to `http://localhost:8000`
2. Click "Start Meeting & Record"
3. **Grant microphone permission** when prompted by your browser
4. Speak naturally - transcription will appear in real-time
5. Click "Stop Meeting" when finished
6. View the generated summary

## Configuration

### Audio Settings (config.yaml)

```yaml
audio:
  sample_rate: 16000  # Required for Whisper compatibility
  channels: 1         # Mono audio
  chunk_size: 1024    # Processing chunk size
  format: "wav"
```

**Note**: The browser version doesn't use `input_device` setting as audio comes from the browser.

### STT Engine Selection

Supported engines:
- `whisper` - OpenAI Whisper (Python)
- `whispercpp` - Whisper.cpp (faster, C++ implementation)
- `vosk` - Offline Vosk engine

Change default engine in `config.yaml`:
```yaml
stt:
  default_engine: "whisper"  # or "whispercpp", "vosk"
```

### Summarization Engine Selection

Supported engines:
- `qwen3` - Qwen 2.5 3B (local, good balance)
- `ollama` - Ollama with various models
- `openai` - OpenAI GPT (requires API key)

Change default engine in `config.yaml`:
```yaml
summarization:
  default_engine: "qwen3"  # or "ollama", "openai"
```

## Browser Compatibility

### Supported Browsers

| Browser | Minimum Version | Notes |
|---------|----------------|-------|
| Chrome | 56+ | Recommended, best performance |
| Firefox | 52+ | Fully supported |
| Edge | 79+ | Chromium-based Edge |
| Safari | 14.1+ | macOS 11+, iOS 14.5+ |
| Opera | 43+ | Chromium-based |

### Required Browser Features

- Web Audio API
- WebSocket support
- MediaDevices API (getUserMedia)
- ES6 JavaScript support

### Checking Browser Support

The application will automatically detect if your browser supports required features and display an error if not compatible.

## Technical Details

### Audio Format

- **Sample Rate**: 16kHz (required for Whisper)
- **Channels**: Mono (1 channel)
- **Format**: 16-bit PCM (Int16)
- **Encoding**: Raw binary over WebSocket

### WebSocket Protocol

#### Audio Streaming (Binary)
```
Client → Server: Int16Array audio chunks (binary)
```

#### Control Messages (JSON)
```javascript
// Transcription result
Server → Client: {
  "type": "transcription",
  "text": "transcribed text",
  "timestamp": 1234567890.123,
  "confidence": 0.95
}

// Audio level
Server → Client: {
  "type": "audio_level",
  "level": 0.42
}

// Connection status
Server → Client: {
  "type": "connected",
  "message": "Audio stream connected",
  "session_id": "meeting_123456"
}

// Error
Server → Client: {
  "type": "error",
  "message": "Error description"
}
```

### Performance Considerations

1. **Network Bandwidth**: Audio streaming requires ~256 Kbps (16kHz * 16-bit)
2. **Latency**: Typical transcription latency is 2-4 seconds
3. **Processing**: Server-side processing, no browser computation needed
4. **Memory**: Minimal browser memory usage (~20-50 MB)

## Troubleshooting

### Microphone Permission Denied

**Problem**: Browser doesn't request or denies microphone access.

**Solutions**:
1. Check browser settings: Settings → Privacy → Microphone
2. Ensure you're using HTTPS or localhost (HTTP only works on localhost)
3. Clear site permissions and reload
4. Try a different browser

### No Transcription Appearing

**Problem**: Audio is being captured but no transcription shows.

**Solutions**:
1. Check server logs for STT engine errors
2. Verify STT model is downloaded and initialized
3. Check audio level meter - ensure audio is being captured
4. Increase speaking volume or improve microphone quality

### WebSocket Connection Failed

**Problem**: Cannot connect to server.

**Solutions**:
1. Verify server is running: `python web_app_browser.py`
2. Check firewall settings
3. Ensure correct port (default: 8000)
4. Check browser console for detailed error messages

### Poor Transcription Quality

**Problem**: Transcriptions are inaccurate.

**Solutions**:
1. Use a better quality microphone
2. Reduce background noise
3. Speak clearly and at a moderate pace
4. Try a larger Whisper model (e.g., medium instead of tiny)
5. Ensure you're using the correct language setting

### High Latency

**Problem**: Transcription appears slowly.

**Solutions**:
1. Switch to WhisperCPP for faster processing
2. Use a smaller model (e.g., tiny or base)
3. Enable GPU/NPU acceleration if available
4. Reduce background processes on server

## Comparison with Server Version

### Browser Version

**Advantages**:
- No PyAudio installation required
- Works on any system with a modern browser
- Users can select their own microphone in browser settings
- No server-side audio device configuration needed
- Better for remote/distributed teams

**Disadvantages**:
- Requires microphone permission from browser
- Depends on network connection for audio streaming
- May have slightly higher latency due to network transmission
- Browser compatibility requirements

### Server Version

**Advantages**:
- Lower latency (direct audio capture)
- No network overhead for audio
- Can capture from professional audio equipment
- Better for local, in-person meetings

**Disadvantages**:
- Requires PyAudio installation (can be complex)
- Audio device configuration can be tricky
- Must have microphone connected to server
- Platform-specific audio driver issues

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/status` | GET | Get system status |
| `/api/meeting/start` | POST | Start a new meeting |
| `/api/meeting/stop` | POST | Stop current meeting |
| `/api/meeting/{session_id}` | GET | Get meeting details |
| `/api/transcribe` | POST | Transcribe uploaded audio file |
| `/api/engines/stt/switch` | POST | Switch STT engine |
| `/api/engines/summarization/switch` | POST | Switch summarization engine |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/audio/{session_id}` | Audio streaming and transcription |

## File Structure

```
/home/amd/MeetingassistantBrowser/
├── web_app_browser.py          # Main FastAPI application
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── README_BROWSER.md           # This file
├── static/
│   ├── css/
│   │   └── style.css           # UI styling
│   └── js/
│       ├── audio-capture.js    # Web Audio API implementation
│       ├── websocket-client.js # WebSocket client
│       └── app.js              # Main application logic
├── templates/
│   └── index.html              # Main web page
├── src/                        # Backend modules
│   ├── stt/                    # Speech-to-text engines
│   ├── summarization/          # Summarization engines
│   ├── config.py               # Configuration loader
│   └── utils/                  # Utility functions
├── data/
│   └── meetings/               # Saved meetings (JSON)
└── models/                     # STT models directory
```

## Development

### Adding New Features

1. **Frontend Changes**: Edit files in `static/js/` and `static/css/`
2. **Backend Changes**: Edit `web_app_browser.py` and modules in `src/`
3. **UI Changes**: Edit `templates/index.html`

### Testing

1. **Browser Console**: Check for JavaScript errors
2. **Server Logs**: Monitor FastAPI logs for backend errors
3. **Network Tab**: Inspect WebSocket frames and messages

### Debugging

Enable debug mode in `config.yaml`:
```yaml
app:
  debug: true

server:
  reload: true  # Auto-reload on code changes
```

## Security Considerations

1. **HTTPS**: Use HTTPS in production for secure WebSocket (WSS)
2. **Authentication**: Add authentication for production deployments
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **Input Validation**: All user inputs are validated
5. **XSS Protection**: HTML escaping prevents XSS attacks

## License

See main project LICENSE file.

## Support

For issues and questions:
1. Check this README first
2. Review troubleshooting section
3. Check server logs for detailed errors
4. Refer to main project documentation

## Changelog

### Version 1.0.0 (Browser)
- Initial browser-based version
- Web Audio API integration
- WebSocket audio streaming
- Real-time transcription display
- Meeting summarization
- Responsive UI design
- Multi-browser support
