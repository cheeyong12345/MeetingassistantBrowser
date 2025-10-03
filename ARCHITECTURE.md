# Architecture Documentation

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ User Input │──▶│ Web Audio API │──▶│ Audio Processing    │  │
│  │ (Microphone│   │ (16kHz, Mono)│   │ (Float32 → Int16)   │  │
│  └────────────┘   └──────────────┘   └──────────────────────┘  │
│                           │                      │               │
│                           ▼                      ▼               │
│                    ┌─────────────────────────────────┐          │
│                    │   WebSocket Client (JS)         │          │
│                    │   - Binary: Audio Data          │          │
│                    │   - JSON: Control Messages      │          │
│                    └─────────────────────────────────┘          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ WebSocket (ws:// or wss://)
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       Server (Backend)                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              FastAPI + WebSocket Server                   │  │
│  │  - Receives binary audio chunks                           │  │
│  │  - Buffers audio data                                     │  │
│  │  - Routes to STT engine                                   │  │
│  └───────────────────┬──────────────────────────────────────┘  │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              STT Manager                                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │  │
│  │  │ Whisper  │  │WhisperCPP│  │  Vosk    │               │  │
│  │  │ (Python) │  │   (C++)  │  │ (Offline)│               │  │
│  │  └──────────┘  └──────────┘  └──────────┘               │  │
│  └───────────────────┬──────────────────────────────────────┘  │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Transcription Result                           │  │
│  │  - Text                                                   │  │
│  │  - Confidence                                             │  │
│  │  - Timestamp                                              │  │
│  └───────────────────┬──────────────────────────────────────┘  │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Meeting Session Manager                        │  │
│  │  - Accumulates transcript                                 │  │
│  │  - Manages meeting lifecycle                              │  │
│  │  - Triggers summarization                                 │  │
│  └───────────────────┬──────────────────────────────────────┘  │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Summarization Manager                            │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐                      │  │
│  │  │ Qwen   │  │ Ollama │  │ OpenAI │                      │  │
│  │  │ (Local)│  │(Local) │  │  (API) │                      │  │
│  │  └────────┘  └────────┘  └────────┘                      │  │
│  └───────────────────┬──────────────────────────────────────┘  │
│                      │                                          │
│                      ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Data Persistence                             │  │
│  │  - JSON files (meetings)                                  │  │
│  │  - SQLite database (optional)                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend (Browser)

#### Audio Capture Module (`audio-capture.js`)
- **Purpose**: Capture audio from user's microphone
- **Technology**: Web Audio API
- **Key Features**:
  - Request microphone permission
  - Create AudioContext at 16kHz sample rate
  - Process audio in real-time chunks
  - Convert Float32 to Int16 PCM
  - Calculate audio levels for visualization

**Audio Processing Flow**:
```javascript
navigator.mediaDevices.getUserMedia()
  ↓
AudioContext (16kHz)
  ↓
MediaStreamSource
  ↓
ScriptProcessorNode (4096 buffer)
  ↓
Float32Array → Int16Array conversion
  ↓
Send to WebSocket
```

#### WebSocket Client Module (`websocket-client.js`)
- **Purpose**: Bidirectional communication with server
- **Protocol**: WebSocket (binary + JSON)
- **Key Features**:
  - Connection management with auto-reconnect
  - Binary audio streaming
  - JSON message handling
  - Heartbeat/ping-pong
  - Error handling and recovery

**Message Types**:
```
Client → Server:
- Binary: Int16Array audio data
- JSON: { type: "ping" }
- JSON: { type: "get_transcript" }

Server → Client:
- JSON: { type: "connected", ... }
- JSON: { type: "transcription", text, confidence, ... }
- JSON: { type: "audio_level", level }
- JSON: { type: "error", message }
```

#### Application Module (`app.js`)
- **Purpose**: Main application logic and UI management
- **Responsibilities**:
  - Orchestrate audio capture and WebSocket
  - Handle user interactions
  - Update UI in real-time
  - Display transcriptions and summaries
  - Error handling and user feedback

### 2. Backend (Server)

#### FastAPI Application (`web_app_browser.py`)
- **Framework**: FastAPI with WebSocket support
- **Key Endpoints**:

**REST API**:
```python
GET  /                      # Main web interface
GET  /api/status            # System status
POST /api/meeting/start     # Start meeting
POST /api/meeting/stop      # Stop meeting
GET  /api/meeting/{id}      # Get meeting details
POST /api/transcribe        # Upload & transcribe file
```

**WebSocket**:
```python
WS /ws/audio/{session_id}   # Audio streaming
```

#### Session Management

**MeetingSession Class**:
```python
class MeetingSession:
    - session_id: str
    - title: str
    - participants: List[str]
    - start_time: datetime
    - end_time: Optional[datetime]
    - transcript: List[Dict]
    - audio_buffer: AudioBuffer
    - summary: Optional[str]
```

**AudioBuffer Class**:
```python
class AudioBuffer:
    - buffer: List[np.ndarray]
    - total_samples: int
    - sample_rate: int = 16000

    Methods:
    - add_chunk(audio_data)
    - get_audio() → np.ndarray
    - clear()
    - get_duration() → float
```

#### STT Integration

**Audio Processing Pipeline**:
```
WebSocket receives Int16 binary
  ↓
Convert to numpy array
  ↓
Normalize to Float32
  ↓
Buffer chunks (3 seconds)
  ↓
Pass to STT engine
  ↓
Receive transcription
  ↓
Send to client via WebSocket
```

**Supported STT Engines**:
1. **Whisper (OpenAI)**
   - Python implementation
   - High accuracy
   - Multiple model sizes
   - GPU/NPU acceleration

2. **WhisperCPP**
   - C++ implementation
   - Faster inference
   - Lower memory usage
   - GGML format models

3. **Vosk**
   - Fully offline
   - Fast and lightweight
   - Lower accuracy
   - Good for privacy

#### Summarization Integration

**Summarization Pipeline**:
```
Meeting ends
  ↓
Collect full transcript
  ↓
Pass to summarization engine
  ↓
Generate summary
  ↓
Save to meeting data
  ↓
Return to client
```

**Supported Engines**:
1. **Qwen 2.5 3B**
   - Local model
   - Good balance
   - NPU/GPU acceleration

2. **Ollama**
   - Multiple models
   - Easy model management
   - Local inference

3. **OpenAI GPT**
   - Cloud API
   - High quality
   - Requires API key

## Data Flow

### Starting a Meeting

```
1. User clicks "Start Meeting"
   ↓
2. Browser sends POST /api/meeting/start
   ↓
3. Server creates MeetingSession
   ↓
4. Server returns session_id
   ↓
5. Browser requests microphone permission
   ↓
6. Browser connects to /ws/audio/{session_id}
   ↓
7. WebSocket connection established
   ↓
8. Browser starts audio capture
   ↓
9. Audio chunks sent to server
```

### Real-time Transcription

```
1. Browser captures audio chunk (4096 samples)
   ↓
2. Convert Float32 → Int16 PCM
   ↓
3. Send via WebSocket (binary)
   ↓
4. Server receives and buffers
   ↓
5. When buffer reaches 3 seconds:
   ↓
6. Pass to STT engine
   ↓
7. STT returns transcription
   ↓
8. Add to meeting transcript
   ↓
9. Send to browser (JSON)
   ↓
10. Browser displays in UI
```

### Stopping a Meeting

```
1. User clicks "Stop Meeting"
   ↓
2. Browser stops audio capture
   ↓
3. Browser disconnects WebSocket
   ↓
4. Browser sends POST /api/meeting/stop
   ↓
5. Server marks meeting as ended
   ↓
6. Server generates summary
   ↓
7. Server saves meeting data
   ↓
8. Server returns meeting details
   ↓
9. Browser displays summary
```

## Performance Considerations

### Browser Performance
- **Memory Usage**: ~20-50 MB
- **CPU Usage**: ~5-15% (audio processing)
- **Network**: ~256 Kbps for audio streaming

### Server Performance
- **Memory**: Depends on STT model size
  - Whisper tiny: ~400 MB
  - Whisper medium: ~1.5 GB
  - Whisper large: ~3 GB
- **CPU**: Heavy during transcription
- **GPU/NPU**: Significant speedup when available

### Latency Breakdown
```
Audio capture: ~10ms
Network transfer: ~50-100ms
Server buffering: ~3000ms (by design)
STT processing: ~500-2000ms
Network return: ~50-100ms
UI update: ~10ms
─────────────────────────────
Total: ~3.6-5.2 seconds
```

## Security Considerations

### Browser Security
1. **HTTPS Required**: For production, use HTTPS for secure WebSocket (WSS)
2. **Microphone Permission**: Browser handles permission prompts
3. **XSS Protection**: HTML escaping prevents injection attacks
4. **CORS**: Configure for production deployments

### Server Security
1. **Input Validation**: All inputs validated and sanitized
2. **Rate Limiting**: Prevent abuse (should implement)
3. **Authentication**: Add for production use
4. **WebSocket Security**: Validate session IDs, implement timeouts

### Data Privacy
1. **Local Processing**: STT and summarization can run entirely locally
2. **No External Services**: Optional (Vosk + Qwen)
3. **Data Storage**: Meetings saved locally, can be encrypted
4. **Audio Retention**: Audio not saved by default (only transcript)

## Scalability

### Single User
- Current design optimized for single user
- One meeting at a time recommended

### Multiple Concurrent Users
To support multiple users:
1. Session isolation (already implemented)
2. Resource pooling for STT engines
3. Queue management for summarization
4. Load balancing for WebSocket connections

### Horizontal Scaling
For high-scale deployments:
1. Separate WebSocket servers
2. Centralized Redis for session state
3. Dedicated STT service workers
4. Message queue (RabbitMQ/Kafka) for processing

## Technology Stack

### Frontend
- Vanilla JavaScript (ES6+)
- Web Audio API
- WebSocket API
- Modern CSS (Grid, Flexbox)

### Backend
- Python 3.10+
- FastAPI 0.104+
- WebSockets library
- NumPy for audio processing
- SQLAlchemy (optional database)

### AI/ML
- OpenAI Whisper
- Whisper.cpp (ggerganov)
- Vosk
- Transformers (HuggingFace)
- PyTorch

## Deployment

### Development
```bash
python web_app_browser.py
# Auto-reload enabled
# Debug mode on
```

### Production
```bash
uvicorn web_app_browser:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --ssl-keyfile key.pem \
  --ssl-certfile cert.pem
```

### Docker (Future)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "web_app_browser:app", "--host", "0.0.0.0"]
```

## Future Enhancements

### Planned Features
1. Multi-user support with authentication
2. Speaker diarization
3. Custom vocabulary/terminology
4. Live collaboration (multiple viewers)
5. Recording playback
6. Export formats (PDF, DOCX, TXT)
7. Language selection UI
8. Advanced audio preprocessing
9. Mobile app version
10. Desktop app (Electron)

### Performance Optimizations
1. AudioWorklet instead of ScriptProcessorNode
2. WebAssembly for audio processing
3. Streaming transcription (partial results)
4. Progressive summarization
5. Client-side caching
6. Service worker for offline support

## Comparison with Server Version

| Aspect | Browser Version | Server Version |
|--------|----------------|----------------|
| Audio Source | Browser mic via Web Audio API | Server mic via PyAudio |
| Installation | No PyAudio needed | Requires PyAudio |
| Complexity | More complex (frontend + backend) | Simpler (backend only) |
| Latency | Higher (~3-5s) | Lower (~1-3s) |
| Use Case | Remote/distributed teams | Local meetings |
| Mic Selection | Browser settings | Config file |
| Cross-platform | Works anywhere | Platform-specific issues |
| Network | Required for audio | Not required |

## Monitoring and Debugging

### Browser DevTools
- Console: JavaScript errors and logs
- Network: WebSocket frames inspection
- Performance: CPU/memory profiling

### Server Logs
- FastAPI access logs
- Custom application logs
- STT engine logs
- Error stack traces

### Metrics to Track
- WebSocket connection uptime
- Audio chunk processing time
- STT transcription latency
- Summarization time
- Memory usage trends
- Error rates

## Conclusion

This architecture provides a robust, scalable solution for browser-based meeting transcription while maintaining compatibility with existing STT and summarization engines. The separation of concerns between frontend audio capture and backend processing allows for flexible deployment and easy maintenance.
