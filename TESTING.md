# Testing Guide

Comprehensive testing guide for Meeting Assistant Browser Version.

## Pre-Testing Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Modern browser available (Chrome, Firefox, Edge, Safari)
- [ ] Working microphone connected
- [ ] Server can bind to port 8000 (or configured port)

## Unit Testing

### Backend Components

#### 1. Configuration Loading
```bash
python -c "from src.config import config; print(config.stt.default_engine)"
```
**Expected**: Should print configured STT engine (e.g., "whisper")

#### 2. STT Manager Initialization
```python
from src.stt import STTManager
from src.config import config

manager = STTManager(config.stt.to_dict())
success = manager.initialize()
print(f"STT Manager initialized: {success}")
print(f"Available engines: {manager.get_available_engines()}")
```
**Expected**:
- `success = True` (or False with clear error message)
- List of available engines

#### 3. Summarization Manager Initialization
```python
from src.summarization import SummarizationManager
from src.config import config

manager = SummarizationManager(config.summarization.to_dict())
success = manager.initialize()
print(f"Summarization Manager initialized: {success}")
print(f"Available engines: {manager.get_available_engines()}")
```
**Expected**:
- `success = True` (or False with clear error message)
- List of available engines

### Frontend Components

#### 1. Browser Compatibility Check
Open browser console and run:
```javascript
console.log('getUserMedia:', !!navigator.mediaDevices?.getUserMedia);
console.log('AudioContext:', !!(window.AudioContext || window.webkitAudioContext));
console.log('WebSocket:', !!window.WebSocket);
```
**Expected**: All should print `true`

#### 2. Audio Capture Initialization
In browser console:
```javascript
const capture = new AudioCapture();
capture.initialize().then(success => {
    console.log('Audio initialized:', success);
});
```
**Expected**: Should prompt for microphone permission and log `true`

## Integration Testing

### Test 1: Server Startup

**Steps**:
1. Start server: `python web_app_browser.py`
2. Observe startup logs

**Expected Output**:
```
============================================================
Meeting Assistant - Browser Version
============================================================
Server: http://localhost:8000
Audio Source: Browser Microphone (Web Audio API)
============================================================
INFO: Started server process
INFO: Waiting for application startup
INFO: STT Manager initialized successfully
INFO: Summarization Manager initialized successfully
INFO: Application startup complete
```

**Pass Criteria**:
- Server starts without errors
- STT manager initializes (or clear error if models missing)
- Port binds successfully

### Test 2: Web Interface Loading

**Steps**:
1. Open browser to `http://localhost:8000`
2. Verify page loads

**Expected**:
- Page loads completely
- No JavaScript errors in console
- System status shows initialization state
- Audio mode shows "Browser Microphone"

**Pass Criteria**:
- All UI elements visible
- No 404 errors for static files
- JavaScript loads without errors

### Test 3: Microphone Permission

**Steps**:
1. Click "Start Meeting & Record"
2. Grant microphone permission when prompted

**Expected**:
- Browser shows permission dialog
- After granting, recording status changes to "Recording"
- Audio level meter starts showing activity

**Pass Criteria**:
- Permission granted successfully
- Audio capture initialized
- WebSocket connects

### Test 4: WebSocket Connection

**Steps**:
1. Start a meeting
2. Open browser DevTools → Network → WS tab
3. Observe WebSocket connection

**Expected**:
- WebSocket connects to `/ws/audio/{session_id}`
- Connection status shows "Connected"
- Binary frames sent periodically (audio data)
- JSON frames received (transcription, audio_level)

**Pass Criteria**:
- Connection established
- No immediate disconnections
- Data flowing both directions

### Test 5: Audio Capture

**Steps**:
1. Start a meeting
2. Speak into microphone
3. Observe audio level meter

**Expected**:
- Audio level bar moves when speaking
- Bar is green/yellow during normal speech
- Bar is red during loud speech
- Bar is gray/low when silent

**Pass Criteria**:
- Audio levels respond to voice
- No constant maximum level (saturation)
- No constant zero level (not capturing)

### Test 6: Real-time Transcription

**Steps**:
1. Start a meeting
2. Speak clearly: "This is a test of the transcription system."
3. Wait 3-5 seconds
4. Check transcript container

**Expected**:
- Transcript container becomes visible
- Transcribed text appears within 5 seconds
- Text is reasonably accurate
- Timestamp shown

**Pass Criteria**:
- Transcription appears
- Text is recognizable
- No constant errors
- Multiple segments can be added

### Test 7: Meeting Stop and Summary

**Steps**:
1. Start a meeting
2. Speak several sentences
3. Wait for transcriptions to appear
4. Click "Stop Meeting"
5. Wait for summary

**Expected**:
- Recording stops
- WebSocket disconnects
- Summary container appears
- Summary is generated (if enough content)
- Meeting saved to `data/meetings/`

**Pass Criteria**:
- Clean shutdown
- Summary generated
- File saved successfully
- No errors in console

## Browser Compatibility Testing

### Test on Each Browser

Test matrix:

| Browser | Version | Audio Capture | WebSocket | Transcription | Summary | Notes |
|---------|---------|---------------|-----------|---------------|---------|-------|
| Chrome | Latest | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | |
| Firefox | Latest | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | |
| Edge | Latest | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | |
| Safari | Latest | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ | |

**Test Procedure**:
1. Open application in browser
2. Complete full meeting workflow
3. Document any issues
4. Mark ✓ or ✗ in table

## Performance Testing

### Test 1: Audio Streaming Performance

**Metrics to Measure**:
- Audio chunk send rate (should be ~10-15/second)
- WebSocket latency (RTT)
- Memory usage in browser
- CPU usage in browser

**Tools**:
- Browser DevTools → Performance tab
- Network tab for WebSocket timing

**Pass Criteria**:
- Consistent audio streaming
- No dropped chunks
- Memory stable (not growing continuously)
- CPU < 20% on average

### Test 2: Transcription Latency

**Measure**:
1. Note when you speak
2. Note when transcription appears
3. Calculate difference

**Expected Latency**:
- Whisper tiny: 1-3 seconds
- Whisper base: 2-4 seconds
- Whisper medium: 3-6 seconds
- WhisperCPP: 1-3 seconds (faster)
- Vosk: 1-2 seconds (fastest)

**Pass Criteria**:
- Latency within expected range
- Consistent (not variable)

### Test 3: Long Meeting Test

**Procedure**:
1. Start a meeting
2. Let it run for 30 minutes
3. Periodically speak
4. Monitor system resources

**Metrics**:
- Browser memory usage
- Server memory usage
- Transcript accuracy over time
- WebSocket stability

**Pass Criteria**:
- No memory leaks
- Connection remains stable
- Transcription quality consistent
- No crashes or disconnections

## Error Handling Testing

### Test 1: Microphone Permission Denied

**Steps**:
1. Start meeting
2. Deny microphone permission

**Expected**:
- Error message displayed
- Clear user guidance
- Meeting doesn't start
- No JavaScript errors

### Test 2: WebSocket Connection Lost

**Steps**:
1. Start meeting
2. Stop server (simulate connection loss)
3. Observe client behavior

**Expected**:
- Error message displayed
- "Disconnected" status shown
- Graceful handling (no crash)
- Reconnect attempts if server restarts

### Test 3: No STT Engine Available

**Steps**:
1. Configure invalid STT engine in config.yaml
2. Start server
3. Attempt to start meeting

**Expected**:
- Server starts with warning
- System status shows "Not Available"
- Clear error message to user
- Guidance on installing models

### Test 4: Network Interruption

**Steps**:
1. Start meeting
2. Disable network briefly (or use DevTools throttling)
3. Re-enable network

**Expected**:
- Connection status updates
- Automatic reconnection attempt
- Session recovers if possible
- Data loss minimized

## Security Testing

### Test 1: HTTPS/WSS (Production)

**Verify**:
- WSS used instead of WS on HTTPS
- No mixed content warnings
- Certificate valid

### Test 2: Input Sanitization

**Test**:
1. Enter XSS attempt in meeting title: `<script>alert('xss')</script>`
2. Submit and check display

**Expected**:
- Text displayed as literal string
- No script execution
- HTML escaped properly

### Test 3: Session Isolation

**Test**:
1. Start meeting A
2. Open new tab, try to connect to meeting A's WebSocket
3. Verify access control

**Expected**:
- Each session isolated
- No cross-session access

## Stress Testing

### Test 1: Rapid Start/Stop

**Procedure**:
1. Start meeting
2. Immediately stop
3. Repeat 10 times quickly

**Expected**:
- No resource leaks
- Clean start/stop each time
- No stuck sessions

### Test 2: Concurrent Meetings (if supported)

**Procedure**:
1. Open multiple browser tabs
2. Start meeting in each
3. Monitor server resources

**Expected**:
- Multiple sessions handled
- Resources allocated properly
- No interference between sessions

## Accessibility Testing

### Test 1: Keyboard Navigation

**Verify**:
- Tab navigation works
- Enter key activates buttons
- Focus indicators visible
- No keyboard traps

### Test 2: Screen Reader

**Test with Screen Reader**:
- Form labels read correctly
- Status updates announced
- Error messages accessible
- Transcript readable

## Test Results Template

```markdown
## Test Session: [Date]

**Tester**: [Name]
**Environment**:
- OS: [Linux/Mac/Windows]
- Browser: [Chrome 120]
- Python: [3.10.12]
- STT Engine: [whisper-medium]

### Results

| Test | Status | Notes |
|------|--------|-------|
| Server Startup | ✓/✗ | |
| Web Interface | ✓/✗ | |
| Microphone Permission | ✓/✗ | |
| WebSocket Connection | ✓/✗ | |
| Audio Capture | ✓/✗ | |
| Real-time Transcription | ✓/✗ | |
| Meeting Stop & Summary | ✓/✗ | |

### Issues Found
1. [Issue description]
2. [Issue description]

### Performance Metrics
- Transcription Latency: [X seconds]
- Memory Usage: [X MB]
- CPU Usage: [X%]

### Recommendations
- [Recommendation]
```

## Automated Testing (Future)

### Unit Tests
```bash
pytest tests/test_audio_buffer.py
pytest tests/test_session_manager.py
pytest tests/test_websocket.py
```

### Integration Tests
```bash
pytest tests/test_integration.py
```

### E2E Tests (Playwright/Selenium)
```bash
pytest tests/test_e2e.py
```

## Continuous Testing Checklist

Before each release:
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Tested on Chrome
- [ ] Tested on Firefox
- [ ] Tested on Edge
- [ ] Performance benchmarks met
- [ ] No memory leaks detected
- [ ] Error handling verified
- [ ] Documentation updated
- [ ] Security scan passed

## Known Issues and Limitations

Document any known issues:
1. [Issue]: [Description] - [Workaround]
2. Safari may require additional user interaction before audio capture
3. Mobile browsers have limited support (use desktop)
4. High latency on slow connections

## Reporting Issues

When reporting issues, include:
1. Steps to reproduce
2. Expected vs actual behavior
3. Browser and version
4. Server logs
5. Browser console logs
6. Screenshots/recordings
7. Configuration (config.yaml)

---

**Testing Status**: Last updated [Date]
