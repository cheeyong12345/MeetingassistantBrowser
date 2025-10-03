- always use sub agent for the task
- always show to-dos first
- always use sub agent for specified task

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

# Project-Specific Instructions

## Browser-Based Meeting Assistant
This is the browser-based version that captures audio from the client's browser microphone instead of server-side audio devices.

### Key Differences from Server Version:
1. **No PyAudio** - Audio capture happens in browser using Web Audio API
2. **WebSocket Audio Streaming** - Binary audio sent from browser to server
3. **Browser Microphone** - Uses client device microphone, not server hardware
4. **Cross-platform** - Works on any device with modern browser

### Important Technical Details:
- Audio format: 16kHz mono 16-bit PCM (Whisper-compatible)
- WebSocket endpoint: ws://localhost:8000/ws/audio
- Binary audio chunks + JSON control messages
- Real-time transcription display in browser

### What NOT to Do:
- DO NOT add PyAudio dependencies
- DO NOT try to capture audio server-side
- DO NOT modify the Web Audio API implementation without understanding browser compatibility
- DO NOT break the WebSocket binary/JSON protocol

### Testing:
- Always test in browser (Chrome, Firefox, or Edge)
- Check browser console for JavaScript errors
- Verify microphone permissions are granted
- Test WebSocket connection status

### Architecture:
```
Browser (Client) ←→ WebSocket ←→ FastAPI Server ←→ STT/Summarization
    ↑                                                        ↑
Web Audio API                                      Whisper/Qwen Engines
```
