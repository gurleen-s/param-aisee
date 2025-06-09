import asyncio
import logging
import json
import time
import tempfile
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime

# Audio processing
import librosa
import numpy as np
from pydub import AudioSegment

# ML for diarization and analysis
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    
import openai

from ..events import Event, EventType, event_bus
from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    concepts: List[str] = None
    
@dataclass 
class MeetingAnalysis:
    current_topics: List[str]
    key_entities: List[str]
    sentiment: str
    suggestions: List[str]
    action_items: List[str]

class MeetingSimulator:
    def __init__(self, io_pool: concurrent.futures.ThreadPoolExecutor, audio_processor=None):
        self.io_pool = io_pool
        self.audio_processor = audio_processor  # Add audio processor integration
        self.is_playing = False
        self.current_meeting: Optional[Dict] = None
        self.segments: List[SpeakerSegment] = []
        self.playback_speed = 1.0
        self.current_position = 0.0
        
        # Analysis state
        self.conversation_buffer: List[str] = []
        self.current_analysis: Optional[MeetingAnalysis] = None
        
        # Playback state for continuous mode
        self.continuous_mode_active = False
        self.playback_task: Optional[asyncio.Task] = None
        
        # Diarization pipeline
        self.diarization_pipeline = None
        if DIARIZATION_AVAILABLE:
            try:
                # Using Hugging Face token from environment if available
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
                )
            except Exception as e:
                logger.warning(f"Failed to load diarization pipeline: {e}")
                self.diarization_pipeline = None
    
    async def upload_meeting_audio(self, file_path: str, meeting_name: str = None) -> Dict[str, Any]:
        """Upload and process meeting audio file"""
        try:
            logger.info(f"Processing meeting audio: {file_path}")
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio file
            audio_info = await self._load_audio_file(file_path)
            
            # Perform speaker diarization
            diarization_result = await self._perform_diarization(file_path)
            
            # Create meeting metadata
            meeting_id = f"meeting_{int(time.time())}"
            self.current_meeting = {
                "id": meeting_id,
                "name": meeting_name or f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "file_path": file_path,
                "duration": audio_info["duration"],
                "sample_rate": audio_info["sample_rate"],
                "diarization": diarization_result,
                "uploaded_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Meeting processed: {meeting_id}")
            
            await event_bus.publish(Event(
                type=EventType.SYSTEM_STATUS,
                action="meeting_uploaded",
                data={
                    "meeting_id": meeting_id,
                    "name": self.current_meeting["name"],
                    "duration": audio_info["duration"],
                    "speakers": len(diarization_result.get("speakers", []))
                }
            ))
            
            return {
                "success": True,
                "meeting_id": meeting_id,
                "duration": audio_info["duration"],
                "speakers": len(diarization_result.get("speakers", [])),
                "message": "Meeting audio processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to process meeting audio: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _load_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Load audio file and extract basic information"""
        def load_audio():
            # Load with librosa for better audio processing
            y, sr = librosa.load(file_path, sr=16000)  # Standardize to 16kHz
            duration = len(y) / sr
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "samples": len(y)
            }
        
        return await asyncio.get_event_loop().run_in_executor(self.io_pool, load_audio)
    
    async def _perform_diarization(self, file_path: str) -> Dict[str, Any]:
        """Perform speaker diarization on audio file"""
        if not DIARIZATION_AVAILABLE or not self.diarization_pipeline:
            logger.warning("Diarization not available, creating mock speakers")
            return await self._create_mock_diarization(file_path)
        
        def run_diarization():
            try:
                # Run diarization
                diarization = self.diarization_pipeline(file_path)
                
                # Convert to our format
                speakers = {}
                segments = []
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_id = f"Speaker_{speaker}"
                    if speaker_id not in speakers:
                        speakers[speaker_id] = {
                            "id": speaker_id,
                            "label": speaker_id,
                            "total_time": 0
                        }
                    
                    segment_duration = turn.end - turn.start
                    speakers[speaker_id]["total_time"] += segment_duration
                    
                    segments.append({
                        "speaker": speaker_id,
                        "start": turn.start,
                        "end": turn.end,
                        "duration": segment_duration
                    })
                
                return {
                    "speakers": list(speakers.values()),
                    "segments": segments,
                    "total_speakers": len(speakers)
                }
                
            except Exception as e:
                logger.error(f"Diarization failed: {e}")
                return {"error": str(e)}
        
        return await asyncio.get_event_loop().run_in_executor(self.io_pool, run_diarization)
    
    async def _create_mock_diarization(self, file_path: str) -> Dict[str, Any]:
        """Create mock speaker diarization for testing"""
        audio_info = await self._load_audio_file(file_path)
        duration = audio_info["duration"]
        
        # Create 2-3 mock speakers with realistic segments
        speakers = [
            {"id": "Speaker_A", "label": "Speaker A", "total_time": duration * 0.4},
            {"id": "Speaker_B", "label": "Speaker B", "total_time": duration * 0.35},
            {"id": "Speaker_C", "label": "Speaker C", "total_time": duration * 0.25}
        ]
        
        # Create mock segments (alternating speakers)
        segments = []
        current_time = 0
        speaker_idx = 0
        
        while current_time < duration:
            segment_duration = min(np.random.uniform(5, 15), duration - current_time)
            segments.append({
                "speaker": speakers[speaker_idx]["id"],
                "start": current_time,
                "end": current_time + segment_duration,
                "duration": segment_duration
            })
            current_time += segment_duration
            speaker_idx = (speaker_idx + 1) % len(speakers)
        
        return {
            "speakers": speakers,
            "segments": segments,
            "total_speakers": len(speakers)
        }
    
    async def start_meeting_playback(self, speed: float = 1.0) -> Dict[str, Any]:
        """Start simulated real-time meeting playback with continuous mode"""
        if not self.current_meeting:
            return {"success": False, "error": "No meeting loaded"}
        
        if self.is_playing:
            return {"success": False, "error": "Meeting already playing"}
        
        # Enable continuous mode in audio processor
        if self.audio_processor:
            try:
                await self.audio_processor.enable_continuous_mode()
                self.continuous_mode_active = True
                logger.info("🎧 Continuous audio processing mode enabled")
            except Exception as e:
                logger.error(f"Failed to enable continuous mode: {e}")
                return {"success": False, "error": f"Failed to enable continuous mode: {e}"}
        else:
            logger.warning("No audio processor available - running in simulation mode only")
        
        self.is_playing = True
        self.playback_speed = speed
        self.current_position = 0.0
        
        logger.info(f"▶️ Starting meeting playback (speed: {speed}x)")
        
        # Start playback task
        self.playback_task = asyncio.create_task(self._playback_loop())
        
        await event_bus.publish(Event(
            type=EventType.SYSTEM_STATUS,
            action="meeting_playback_started",
            data={
                "meeting_id": self.current_meeting["id"],
                "speed": speed,
                "duration": self.current_meeting["duration"],
                "continuous_mode": self.continuous_mode_active
            }
        ))
        
        return {"success": True, "message": "Meeting playback started"}
    
    async def stop_meeting_playback(self) -> Dict[str, Any]:
        """Stop meeting playback and disable continuous mode"""
        if not self.is_playing:
            return {"success": False, "error": "No meeting playing"}
        
        self.is_playing = False
        
        # Cancel playback task
        if self.playback_task and not self.playback_task.done():
            self.playback_task.cancel()
            try:
                await self.playback_task
            except asyncio.CancelledError:
                pass
        
        # Disable continuous mode in audio processor
        if self.audio_processor and self.continuous_mode_active:
            try:
                await self.audio_processor.disable_continuous_mode()
                self.continuous_mode_active = False
                logger.info("🎧 Continuous audio processing mode disabled")
            except Exception as e:
                logger.error(f"Failed to disable continuous mode: {e}")
        
        logger.info("⏹️ Meeting playback stopped")
        
        await event_bus.publish(Event(
            type=EventType.SYSTEM_STATUS,
            action="meeting_playback_stopped",
            data={
                "meeting_id": self.current_meeting["id"] if self.current_meeting else None,
                "continuous_mode": self.continuous_mode_active
            }
        ))
        
        return {"success": True, "message": "Meeting playback stopped"}
    
    async def _playback_loop(self):
        """Main playback loop that simulates real-time meeting"""
        try:
            meeting = self.current_meeting
            duration = meeting["duration"]
            segments = meeting["diarization"]["segments"]
            
            logger.info(f"🎬 Starting playback loop for {duration:.1f}s meeting")
            
            # Process each segment in real-time
            for segment in segments:
                if not self.is_playing:
                    break
                
                # Wait until we reach this segment's start time
                while self.current_position < segment["start"] and self.is_playing:
                    await asyncio.sleep(0.1 / self.playback_speed)
                    self.current_position += 0.1
                
                if not self.is_playing:
                    break
                
                # Process this segment
                await self._process_segment(segment)
                
                # Advance position to end of segment
                self.current_position = segment["end"]
            
            # Meeting finished
            if self.is_playing:
                self.is_playing = False
                await event_bus.publish(Event(
                    type=EventType.SYSTEM_STATUS,
                    action="meeting_playback_finished",
                    data={"meeting_id": meeting["id"]}
                ))
                
        except Exception as e:
            logger.error(f"Playback loop error: {e}")
            self.is_playing = False
    
    async def _process_segment(self, segment: Dict[str, Any]):
        """Process an individual speaker segment with utterance simulation"""
        try:
            # Extract audio for this segment
            audio_text = await self._transcribe_segment(segment)
            
            if not audio_text.strip():
                return
            
            # Create speaker segment
            speaker_segment = SpeakerSegment(
                speaker_id=segment["speaker"],
                start_time=segment["start"],
                end_time=segment["end"],
                text=audio_text
            )
            
            # Simulate utterance event for continuous mode integration
            await event_bus.publish(Event(
                type=EventType.AUDIO_EVENT,
                action="utterance_ready",
                data={
                    "speaker": segment["speaker"],
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": audio_text,
                    "duration": segment["duration"],
                    "simulation": True,  # Mark as simulation
                    "meeting_id": self.current_meeting["id"]
                }
            ))
            
            # Also publish transcript segment event for UI
            await event_bus.publish(Event(
                type=EventType.AUDIO_EVENT,
                action="meeting_transcript_segment",
                data={
                    "speaker": segment["speaker"],
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": audio_text,
                    "duration": segment["duration"],
                    "meeting_id": self.current_meeting["id"]
                }
            ))
            
            # Add to conversation buffer for analysis
            self.conversation_buffer.append(f"{segment['speaker']}: {audio_text}")
            
            # Analyze conversation every few segments
            if len(self.conversation_buffer) >= 3:
                await self._analyze_conversation()
            
            # Simulate real-time playback timing
            segment_duration = segment["duration"]
            await asyncio.sleep(segment_duration / self.playback_speed)
            
        except Exception as e:
            logger.error(f"Error processing segment: {e}")
    
    async def _transcribe_segment(self, segment: Dict[str, Any]) -> str:
        """Transcribe audio segment using MLX Whisper"""
        try:
            def extract_and_transcribe():
                # Load the audio file
                y, sr = librosa.load(
                    self.current_meeting["file_path"],
                    sr=16000,
                    offset=segment["start"],
                    duration=segment["duration"]
                )
                
                # Use MLX Whisper for transcription
                import mlx_whisper
                result = mlx_whisper.transcribe(
                    y.astype(np.float32),
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
                )
                
                return result["text"].strip()
            
            return await asyncio.get_event_loop().run_in_executor(
                self.io_pool, extract_and_transcribe
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Return mock text for testing
            speaker_name = segment["speaker"].replace("_", " ")
            return f"This is a sample statement from {speaker_name} lasting {segment['duration']:.1f} seconds."
    
    async def _analyze_conversation(self):
        """Analyze current conversation buffer and generate insights"""
        try:
            if len(self.conversation_buffer) < 2:
                return
            
            # Combine recent conversation context
            context = "\n".join(self.conversation_buffer[-5:])  # Last 5 segments
            
            # Generate analysis using OpenAI
            analysis = await self._generate_conversation_analysis(context)
            
            if analysis:
                self.current_analysis = analysis
                
                # Publish analysis event
                await event_bus.publish(Event(
                    type=EventType.SYSTEM_STATUS,
                    action="meeting_analysis_update",
                    data={
                        "current_topics": analysis.current_topics,
                        "key_entities": analysis.key_entities,
                        "sentiment": analysis.sentiment,
                        "suggestions": analysis.suggestions,
                        "action_items": analysis.action_items,
                        "conversation_context": context
                    }
                ))
            
        except Exception as e:
            logger.error(f"Conversation analysis failed: {e}")
    
    async def _generate_conversation_analysis(self, context: str) -> Optional[MeetingAnalysis]:
        """Generate conversation analysis using OpenAI"""
        try:
            client = openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
            )
            
            prompt = f"""Analyze this meeting conversation segment and provide insights:

{context}

Provide analysis in JSON format:
{{
    "current_topics": ["topic1", "topic2"],
    "key_entities": ["entity1", "entity2"],
    "sentiment": "positive/neutral/negative",
    "suggestions": ["suggestion1", "suggestion2"],
    "action_items": ["action1", "action2"]
}}

Focus on:
- Main topics being discussed
- Important people, companies, or concepts mentioned
- Overall sentiment/mood
- Helpful suggestions for moving the conversation forward
- Concrete action items that emerge
"""
            
            response = await client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert meeting analyst. Provide concise, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            analysis_data = json.loads(analysis_text)
            
            return MeetingAnalysis(
                current_topics=analysis_data.get("current_topics", []),
                key_entities=analysis_data.get("key_entities", []),
                sentiment=analysis_data.get("sentiment", "neutral"),
                suggestions=analysis_data.get("suggestions", []),
                action_items=analysis_data.get("action_items", [])
            )
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return None
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current meeting playback status"""
        if not self.current_meeting:
            return {"status": "no_meeting", "message": "No meeting loaded"}
        
        return {
            "status": "playing" if self.is_playing else "paused",
            "meeting": {
                "id": self.current_meeting["id"],
                "name": self.current_meeting["name"],
                "duration": self.current_meeting["duration"]
            },
            "playback": {
                "current_position": self.current_position,
                "speed": self.playback_speed,
                "progress": (self.current_position / self.current_meeting["duration"]) * 100
            },
            "continuous_mode": {
                "active": self.continuous_mode_active,
                "audio_processor_available": self.audio_processor is not None
            },
            "analysis": {
                "current_topics": self.current_analysis.current_topics if self.current_analysis else [],
                "key_entities": self.current_analysis.key_entities if self.current_analysis else [],
                "suggestions": self.current_analysis.suggestions if self.current_analysis else []
            }
        }
    
    async def handle_utterance_event(self, event_data: Dict[str, Any]):
        """Handle utterance events from continuous mode audio processing"""
        try:
            # This method can be called when real utterances are detected
            # during meeting playback to integrate with live audio processing
            
            if not self.is_playing:
                return
            
            logger.info(f"🎤 Live utterance detected during meeting playback: {event_data.get('text', '')[:50]}...")
            
            # Publish live utterance event
            await event_bus.publish(Event(
                type=EventType.AUDIO_EVENT,
                action="live_utterance_during_meeting",
                data={
                    "text": event_data.get("text", ""),
                    "timestamp": time.time(),
                    "meeting_id": self.current_meeting["id"],
                    "live_input": True
                }
            ))
            
            # Optionally pause meeting playback for live interaction
            # This could be configurable based on user preferences
            
        except Exception as e:
            logger.error(f"Error handling utterance event: {e}") 