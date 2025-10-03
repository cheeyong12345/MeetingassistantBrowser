# File Index - Meeting Assistant Browser Version

Complete listing of all project files with descriptions.

## Core Application Files

| File | Size | Description |
|------|------|-------------|
| `web_app_browser.py` | 18 KB | Main FastAPI application with WebSocket support |
| `config.yaml` | 2.1 KB | Configuration file (STT, summarization, server) |
| `requirements.txt` | 680 B | Python dependencies (PyAudio removed) |
| `setup.sh` | 1.8 KB | Automated setup script |
| `.gitignore` | 608 B | Git ignore patterns |

## Documentation (42 KB total)

| File | Size | Description |
|------|------|-------------|
| `README_BROWSER.md` | 11 KB | Complete user guide and reference |
| `QUICKSTART.md` | 4.0 KB | 5-minute getting started guide |
| `ARCHITECTURE.md` | 15 KB | Technical architecture documentation |
| `TESTING.md` | 12 KB | Comprehensive testing guide |
| `PROJECT_SUMMARY.md` | 9.5 KB | Project overview and summary |
| `FILES.md` | This file | File index and descriptions |

## Frontend Files (28 KB total)

### HTML Templates
| File | Size | Description |
|------|------|-------------|
| `templates/index.html` | 7.0 KB | Single-page web application |

### JavaScript (27 KB)
| File | Size | Description |
|------|------|-------------|
| `static/js/audio-capture.js` | 6.5 KB | Web Audio API microphone capture |
| `static/js/websocket-client.js` | 7.2 KB | WebSocket client with reconnection |
| `static/js/app.js` | 14 KB | Main application logic and UI |

### CSS
| File | Size | Description |
|------|------|-------------|
| `static/css/style.css` | 7.7 KB | Modern responsive styling |

## Backend Source Files

### Main Modules
| File | Lines | Description |
|------|-------|-------------|
| `src/meeting.py` | 650 | Meeting orchestrator (copied) |
| `src/config.py` | 100 | Configuration loader (copied) |
| `src/config_validator.py` | 400 | Config validation (copied) |
| `src/exceptions.py` | 250 | Custom exceptions (copied) |

### Speech-to-Text (src/stt/)
| File | Lines | Description |
|------|-------|-------------|
| `src/stt/base.py` | 63 | STT engine base class |
| `src/stt/manager.py` | 500 | STT engine manager |
| `src/stt/whisper_engine.py` | 250 | OpenAI Whisper engine |
| `src/stt/whispercpp_engine.py` | 400 | Whisper.cpp engine |
| `src/stt/vosk_engine.py` | 200 | Vosk engine |

### Summarization (src/summarization/)
| File | Lines | Description |
|------|-------|-------------|
| `src/summarization/base.py` | 50 | Summarization base class |
| `src/summarization/manager.py` | 300 | Summarization manager |
| `src/summarization/qwen_engine.py` | 250 | Qwen model engine |
| `src/summarization/ollama_engine.py` | 150 | Ollama integration |

### Audio (src/audio/)
| File | Lines | Description |
|------|-------|-------------|
| `src/audio/recorder.py` | 300 | Audio recorder (not used in browser version) |

### Utilities (src/utils/)
| File | Lines | Description |
|------|-------|-------------|
| `src/utils/logger.py` | 100 | Logging configuration |
| `src/utils/hardware.py` | 200 | Hardware detection |
| `src/utils/npu_acceleration.py` | 150 | NPU acceleration utilities |
| `src/utils/eswin_npu.py` | 200 | Eswin NPU support |

## Runtime Directories

| Directory | Purpose |
|-----------|---------|
| `data/meetings/` | Saved meeting JSON files |
| `data/temp/` | Temporary upload files |
| `models/` | STT model files (user downloads) |
| `venv/` | Python virtual environment |

## File Statistics

### Code Distribution
```
Backend Python:   ~500 lines (web_app_browser.py)
Frontend JS:      ~800 lines (3 files)
Frontend HTML:    ~200 lines
Frontend CSS:     ~400 lines
Documentation:    ~15,000 words
Total Project:    ~2,500 lines of original code
```

### Language Breakdown
```
Python:       60%
JavaScript:   30%
HTML/CSS:     10%
```

### Documentation Coverage
```
Total docs:   42 KB
Code:         ~50 KB
Ratio:        Nearly 1:1 (excellent documentation)
```

## Key Implementation Files

### Critical Path for Understanding

1. **Start Here**: `QUICKSTART.md`
2. **User Guide**: `README_BROWSER.md`
3. **Main Server**: `web_app_browser.py`
4. **Frontend Logic**: `static/js/app.js`
5. **Audio Capture**: `static/js/audio-capture.js`
6. **WebSocket**: `static/js/websocket-client.js`
7. **UI**: `templates/index.html`
8. **Architecture**: `ARCHITECTURE.md`

### Dependencies from Original Project

These files are copied unchanged from `/home/amd/Meetingassistant/`:
- All of `src/` (except no changes needed)
- `config.yaml`
- `models/` structure

### New Files Created for Browser Version

- `web_app_browser.py` (NEW)
- All files in `static/` (NEW)
- All files in `templates/` (NEW)
- All documentation files (NEW)
- `setup.sh` (NEW)
- `.gitignore` (NEW)

## File Relationships

```
web_app_browser.py
├── Uses: src/stt/manager.py
├── Uses: src/summarization/manager.py
├── Uses: src/config.py
├── Serves: templates/index.html
└── Serves: static/*

templates/index.html
├── Loads: static/css/style.css
├── Loads: static/js/audio-capture.js
├── Loads: static/js/websocket-client.js
└── Loads: static/js/app.js

static/js/app.js
├── Uses: AudioCapture (audio-capture.js)
└── Uses: WebSocketClient (websocket-client.js)
```

## Total Project Size

```
Source Code:       ~75 KB
Documentation:     ~42 KB
Dependencies:      ~2 GB (when installed)
Models:            ~500 MB - 3 GB (user downloads)
Total (minimal):   ~120 KB (without venv/models)
```

## Maintenance Files

| File | Purpose |
|------|---------|
| `.gitignore` | Exclude venv, models, data from git |
| `requirements.txt` | Dependency management |
| `setup.sh` | Automated environment setup |
| `config.yaml` | Runtime configuration |

## Testing Files

Current status: Testing framework documented in `TESTING.md`

Future additions (recommended):
```
tests/
├── test_audio_buffer.py
├── test_session_manager.py
├── test_websocket.py
├── test_integration.py
└── test_e2e.py
```

## Build/Deploy Files

Current:
- Manual deployment via `setup.sh`

Future additions (recommended):
```
Dockerfile           # Container deployment
docker-compose.yml   # Multi-service orchestration
.env.example         # Environment variables template
nginx.conf           # Reverse proxy config
systemd/             # System service files
```

---

**File Count**: 
- Python: 25 files
- JavaScript: 3 files
- HTML: 1 file
- CSS: 1 file
- Markdown: 6 files
- Config: 2 files
- Total: 38 core files (excluding __pycache__)

**Last Updated**: October 4, 2025
