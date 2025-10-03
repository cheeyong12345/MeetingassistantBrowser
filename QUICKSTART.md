# Quick Start Guide

Get up and running with Meeting Assistant (Browser Version) in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- Modern web browser (Chrome, Firefox, Edge, or Safari)
- Working microphone

## Installation (3 steps)

### Option 1: Automatic Setup (Recommended)

```bash
cd /home/amd/MeetingassistantBrowser
./setup.sh
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Start the server
python web_app_browser.py
```

You should see:
```
============================================================
Meeting Assistant - Browser Version
============================================================
Server: http://localhost:8000
Audio Source: Browser Microphone (Web Audio API)
============================================================
```

## Using the Application

1. **Open your browser** and go to `http://localhost:8000`

2. **Enter meeting details** (optional):
   - Meeting title (e.g., "Team Standup")
   - Participants (e.g., "John, Jane, Alice")

3. **Click "Start Meeting & Record"**
   - Your browser will ask for microphone permission
   - **Click "Allow"** to grant access

4. **Start speaking**
   - Watch the audio level meter to confirm your mic is working
   - Transcription will appear in real-time below

5. **Click "Stop Meeting"** when finished
   - A summary will be generated automatically
   - Meeting data is saved to `data/meetings/`

## Troubleshooting

### No microphone permission prompt?

- Check your browser settings: Settings → Privacy → Microphone
- Make sure you're on `localhost` or `https://`

### No transcription appearing?

- Check the audio level meter - is it showing activity?
- Look at the terminal for any error messages
- Verify STT engine initialized successfully

### "STT Engine: Not Available"?

You need to download a speech-to-text model first:

**Option A: Use Whisper (easiest)**
```bash
# Models download automatically on first use
# Just start the app and begin a meeting!
```

**Option B: Use Whisper.cpp (faster)**
```bash
# Download a model (medium recommended)
mkdir -p models
cd models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
```

**Option C: Use Vosk (offline)**
```bash
# Download from https://alphacephei.com/vosk/models
# Extract to models/vosk-model-en-us-0.22/
```

Then edit `config.yaml` to set your preferred engine.

## Configuration

### Change STT Engine

Edit `config.yaml`:

```yaml
stt:
  default_engine: "whisper"  # or "whispercpp", "vosk"
  engines:
    whisper:
      model_size: "medium"  # tiny, base, small, medium, large
      language: "auto"
      device: "auto"
```

### Change Server Port

Edit `config.yaml`:

```yaml
server:
  host: "localhost"
  port: 8000  # Change to desired port
```

Or use environment variables:
```bash
export MEETING_ASSISTANT_PORT=9000
python web_app_browser.py
```

## What's Different from Server Version?

| Feature | Browser Version | Server Version |
|---------|----------------|----------------|
| Audio Source | Browser microphone | Server microphone |
| Installation | No PyAudio needed | Requires PyAudio |
| Mic Selection | Browser settings | Config file |
| Best For | Remote teams | Local meetings |

## Next Steps

- Read full documentation: `README_BROWSER.md`
- Customize configuration: `config.yaml`
- Try different STT engines for speed/accuracy
- Enable GPU acceleration for faster processing

## Need Help?

- Check `README_BROWSER.md` for detailed documentation
- Review troubleshooting section above
- Check server logs for error messages
- Ensure your browser is supported (Chrome 56+, Firefox 52+, etc.)

---

**Enjoy your automated meeting notes!**
