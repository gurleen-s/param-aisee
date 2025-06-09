import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from enum import Enum


class EventType(Enum):
    # Core system events
    SYSTEM_STATUS = "system_status"        
    AUDIO_EVENT = "audio_event"            
    LLM_EVENT = "llm_event"               
    VISION_EVENT = "vision_event"          
    TTS_EVENT = "tts_event"               
    TOOL_EVENT = "tool_event"             # Tool execution events
    VOICE_CONTROL = "voice_control"       # Voice dictation control events
    CAMERA_CONTROL = "camera_control"     # Camera capture control events
    MEETING_EVENT = "meeting_event"       # Meeting simulation events
    ERROR = "error"                       


@dataclass
class Event:
    type: EventType
    action: str  # New: specific action within the event type
    data: Any = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "action": self.action,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class EventBus:
    def __init__(self, max_size: int = 100):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._subscribers: list[asyncio.Queue] = []
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers"""
        try:
            # Add to main queue (non-blocking)
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Remove oldest event if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass
        
        # Notify all subscribers
        dead_subscribers = []
        for subscriber_queue in self._subscribers:
            try:
                subscriber_queue.put_nowait(event)
            except asyncio.QueueFull:
                # Subscriber can't keep up, remove them
                dead_subscribers.append(subscriber_queue)
        
        # Clean up dead subscribers
        for dead_sub in dead_subscribers:
            self._subscribers.remove(dead_sub)
    
    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to events, returns a queue of events"""
        subscriber_queue = asyncio.Queue(maxsize=50)
        self._subscribers.append(subscriber_queue)
        return subscriber_queue
    
    async def get_event(self) -> Event:
        """Get the next event from the main queue"""
        return await self._queue.get()
    
    def get_event_nowait(self) -> Optional[Event]:
        """Get an event without waiting"""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


# Global event bus instance
event_bus = EventBus()
