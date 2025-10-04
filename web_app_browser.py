#!/usr/bin/env python3
"""
Meeting Assistant Browser Version - Web Interface
Captures audio from browser microphone instead of server-side audio device
"""

import sys
import os
from pathlib import Path
import json
import asyncio
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, UploadFile, File
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    import aiofiles

    from src.config import config
    from src.stt import STTManager
    from src.summarization import SummarizationManager
    from src.utils.logger import get_logger

    DEPENDENCIES_AVAILABLE = True

except ImportError as e:
    print(f"Missing web dependencies: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Meeting Assistant (Browser)", version="1.0.0")

# Configure CORS for remote browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local network access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global state
stt_manager: Optional[STTManager] = None
summarization_manager: Optional[SummarizationManager] = None
active_meetings: Dict[str, Dict[str, Any]] = {}


class AudioBuffer:
    """Buffer for incoming audio chunks from browser"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.buffer = []
        self.total_samples = 0

    def add_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to buffer"""
        self.buffer.append(audio_data)
        self.total_samples += len(audio_data)

    def get_audio(self) -> np.ndarray:
        """Get all buffered audio as single array"""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.buffer)

    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.total_samples = 0

    def get_duration(self) -> float:
        """Get duration in seconds"""
        return self.total_samples / self.sample_rate


class MeetingSession:
    """Manages a single meeting session"""

    def __init__(self, session_id: str, title: str, participants: List[str]):
        self.session_id = session_id
        self.title = title or f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.participants = participants
        self.start_time = datetime.now()
        self.end_time = None
        self.transcript = []
        self.audio_buffer = AudioBuffer()
        self.real_time_transcript = ""
        self.summary = None

    def add_transcription(self, text: str, timestamp: Optional[float] = None):
        """Add transcription segment"""
        if timestamp is None:
            timestamp = time.time()

        self.transcript.append({
            "text": text,
            "timestamp": timestamp,
            "time_str": datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        })
        self.real_time_transcript += " " + text

    def get_full_transcript(self) -> str:
        """Get full transcript text"""
        return " ".join([seg["text"] for seg in self.transcript])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "participants": self.participants,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "transcript": self.transcript,
            "summary": self.summary,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }

    def save(self, data_dir: Path):
        """Save meeting data to file"""
        meeting_file = data_dir / "meetings" / f"{self.session_id}.json"
        meeting_file.parent.mkdir(parents=True, exist_ok=True)

        with open(meeting_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Meeting saved: {meeting_file}")


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific session"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections"""
        for session_id, connection in list(self.active_connections.items()):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global stt_manager, summarization_manager

    logger.info("Initializing Meeting Assistant (Browser Version)")

    # Initialize STT manager
    try:
        stt_manager = STTManager(config.stt.to_dict())
        # Manager initialization happens in __init__, no separate initialize() call needed
        logger.info("STT Manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize STT Manager: {e}")
        stt_manager = None

    # Initialize summarization manager
    try:
        summarization_manager = SummarizationManager(config.summarization.to_dict())
        # Manager initialization happens in __init__, no separate initialize() call needed
        logger.info("Summarization Manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Summarization Manager: {e}")
        summarization_manager = None

    # Create necessary directories
    Path("data/meetings").mkdir(parents=True, exist_ok=True)
    Path("data/temp").mkdir(parents=True, exist_ok=True)

    logger.info("Meeting Assistant initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Meeting Assistant")

    # Save any active meetings
    for session_id, meeting in active_meetings.items():
        try:
            meeting.save(Path("data"))
        except Exception as e:
            logger.error(f"Error saving meeting {session_id}: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page"""
    status = {
        "stt": stt_manager.get_current_engine_info() if stt_manager else {"initialized": False},
        "summarization": summarization_manager.get_current_engine_info() if summarization_manager else {"initialized": False}
    }

    return templates.TemplateResponse("index.html", {
        "request": request,
        "status": status,
        "version": "1.0.0 (Browser)",
        "audio_mode": "Browser Microphone"
    })


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "stt": stt_manager.get_current_engine_info() if stt_manager else {"initialized": False},
        "summarization": summarization_manager.get_current_engine_info() if summarization_manager else {"initialized": False},
        "active_meetings": len(active_meetings),
        "audio_mode": "browser",
        "available_engines": {
            "stt": stt_manager.get_available_engines() if stt_manager else [],
            "summarization": summarization_manager.get_available_engines() if summarization_manager else []
        }
    }


@app.post("/api/meeting/start")
async def start_meeting(title: str = Form(None), participants: str = Form(None)):
    """Start a new meeting"""
    try:
        # Generate session ID
        session_id = f"meeting_{int(time.time() * 1000)}"

        # Parse participants
        participant_list = []
        if participants:
            participant_list = [p.strip() for p in participants.split(',') if p.strip()]

        # Create meeting session
        meeting = MeetingSession(session_id, title, participant_list)
        active_meetings[session_id] = meeting

        logger.info(f"Meeting started: {session_id} - {meeting.title}")

        return {
            "success": True,
            "session_id": session_id,
            "title": meeting.title,
            "start_time": meeting.start_time.isoformat()
        }

    except Exception as e:
        logger.error(f"Error starting meeting: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/meeting/stop")
async def stop_meeting(session_id: str = Form(...)):
    """Stop a meeting and generate summary"""
    try:
        if session_id not in active_meetings:
            return {"success": False, "error": "Meeting not found"}

        meeting = active_meetings[session_id]
        meeting.end_time = datetime.now()

        # Generate summary if we have transcript
        if meeting.get_full_transcript() and summarization_manager:
            try:
                summary_result = summarization_manager.summarize(meeting.get_full_transcript())
                meeting.summary = summary_result.get("summary", "")
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                meeting.summary = "Summary generation failed"

        # Save meeting
        meeting.save(Path("data"))

        # Get meeting data before removing
        meeting_data = meeting.to_dict()

        # Remove from active meetings
        del active_meetings[session_id]

        logger.info(f"Meeting stopped: {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "meeting": meeting_data
        }

    except Exception as e:
        logger.error(f"Error stopping meeting: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/meeting/{session_id}")
async def get_meeting(session_id: str):
    """Get meeting details"""
    if session_id in active_meetings:
        meeting = active_meetings[session_id]
        return {
            "success": True,
            "active": True,
            "meeting": meeting.to_dict()
        }
    else:
        # Try to load from file
        meeting_file = Path("data/meetings") / f"{session_id}.json"
        if meeting_file.exists():
            with open(meeting_file, 'r') as f:
                meeting_data = json.load(f)
            return {
                "success": True,
                "active": False,
                "meeting": meeting_data
            }
        else:
            return {"success": False, "error": "Meeting not found"}


@app.websocket("/ws/audio/{session_id}")
async def websocket_audio(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for receiving audio from browser"""
    await manager.connect(session_id, websocket)

    # Check if meeting exists
    if session_id not in active_meetings:
        await websocket.send_json({
            "type": "error",
            "message": "Meeting not found. Please start a meeting first."
        })
        await websocket.close()
        return

    meeting = active_meetings[session_id]
    audio_buffer = []
    buffer_duration = 0
    target_buffer_duration = 3.0  # Process every 3 seconds of audio

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Audio stream connected",
            "session_id": session_id
        })

        while True:
            # Receive message (can be binary audio or JSON control)
            message = await websocket.receive()

            if "bytes" in message:
                # Binary audio data
                audio_bytes = message["bytes"]

                # Convert bytes to numpy array (assuming 16-bit PCM from browser)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                # Convert to float32 and normalize
                audio_float = audio_array.astype(np.float32) / 32768.0

                # Add to buffer
                audio_buffer.append(audio_float)
                meeting.audio_buffer.add_chunk(audio_float)
                buffer_duration = len(np.concatenate(audio_buffer)) / 16000.0

                # Process when we have enough audio
                if buffer_duration >= target_buffer_duration:
                    audio_data = np.concatenate(audio_buffer)
                    audio_buffer = []

                    # Transcribe with STT
                    if stt_manager:
                        try:
                            result = stt_manager.transcribe_stream(audio_data)

                            if result and result.get("text"):
                                text = result["text"].strip()

                                if text:
                                    # Add to meeting transcript
                                    meeting.add_transcription(text)

                                    # Send transcription to browser
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "text": text,
                                        "timestamp": time.time(),
                                        "confidence": result.get("confidence", 0.0)
                                    })

                                    logger.debug(f"Transcribed: {text}")

                        except Exception as e:
                            logger.error(f"Transcription error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Transcription error: {str(e)}"
                            })

                # Send audio level for visualization
                if len(audio_float) > 0:
                    audio_level = float(np.abs(audio_float).mean())
                    await websocket.send_json({
                        "type": "audio_level",
                        "level": audio_level
                    })

            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif msg_type == "get_transcript":
                        await websocket.send_json({
                            "type": "transcript",
                            "text": meeting.get_full_transcript(),
                            "segments": meeting.transcript
                        })

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message['text']}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        manager.disconnect(session_id)

    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        manager.disconnect(session_id)
        try:
            await websocket.close()
        except:
            pass


@app.post("/api/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    if not stt_manager:
        return {"success": False, "error": "STT engine not initialized"}

    try:
        # Save uploaded file temporarily
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file = temp_dir / f"upload_{int(time.time())}_{file.filename}"

        async with aiofiles.open(temp_file, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Transcribe
        result = stt_manager.transcribe(str(temp_file))

        # Clean up temp file
        temp_file.unlink(missing_ok=True)

        return {
            "success": True,
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0)
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/engines/stt/switch")
async def switch_stt_engine(engine: str = Form(...)):
    """Switch STT engine"""
    if not stt_manager:
        return {"success": False, "error": "STT manager not initialized"}

    try:
        success = stt_manager.switch_engine(engine)
        return {"success": success, "engine": engine}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/engines/summarization/switch")
async def switch_summarization_engine(engine: str = Form(...)):
    """Switch summarization engine"""
    if not summarization_manager:
        return {"success": False, "error": "Summarization manager not initialized"}

    try:
        success = summarization_manager.switch_engine(engine)
        return {"success": success, "engine": engine}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Check for environment variable overrides
    host = os.getenv('MEETING_ASSISTANT_HOST', config.server.host)
    port = int(os.getenv('MEETING_ASSISTANT_PORT', config.server.port))

    print("=" * 60)
    print("Meeting Assistant - Browser Version")
    print("=" * 60)
    print(f"Server: http://{host}:{port}")
    print("Audio Source: Browser Microphone (Web Audio API)")
    print("=" * 60)

    uvicorn.run(
        "web_app_browser:app",
        host=host,
        port=port,
        reload=True
    )
