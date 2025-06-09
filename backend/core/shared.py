"""Shared resources and dependency injection container"""
import asyncio
import concurrent.futures
from typing import Optional


class SharedResources:
    """Container for shared system resources"""
    
    def __init__(self):
        # Shared thread pools
        self.cpu_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.io_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        
        # Event loop reference
        self.loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def initialize(self):
        """Initialize shared resources"""
        self.loop = asyncio.get_running_loop()
        
        # Create specialized thread pools
        self.cpu_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix="cpu-"
        )
        self.io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="io-"
        )
    
    async def cleanup(self):
        """Clean up resources"""
        if self.cpu_pool:
            self.cpu_pool.shutdown(wait=True)
            self.cpu_pool = None
        
        if self.io_pool:
            self.io_pool.shutdown(wait=True)
            self.io_pool = None


class DependencyContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self.shared = SharedResources()
        self._audio_processor = None
        self._vision_processor = None
        self._video_recorder = None
        self._tool_registry = None
        self._llm_processor = None
        self._task_manager = None
        self._meeting_simulator = None
    
    async def initialize(self):
        """Initialize all dependencies"""
        await self.shared.initialize()
        
        # Import here to avoid circular imports
        from .audio import AudioProcessor
        from .vision import VisionProcessor
        from .video_capture import VideoRecorder
        from .tools import ToolRegistry
        from .llm import LLMProcessor
        from .tasks import TaskManager
        from .meeting_simulator import MeetingSimulator
        
        # Create instances with dependencies
        self._audio_processor = AudioProcessor(
            cpu_pool=self.shared.cpu_pool,
            loop=self.shared.loop
        )
        
        self._vision_processor = VisionProcessor()
        
        # Video recorder depends on vision processor
        self._video_recorder = VideoRecorder(self._vision_processor)
        
        # Tool registry depends on vision processor and video recorder
        self._tool_registry = ToolRegistry(
            vision_processor=self._vision_processor,
            video_recorder=self._video_recorder
        )
        
        # LLM processor now includes tool registry and audio processor
        self._llm_processor = LLMProcessor(
            io_pool=self.shared.io_pool,
            tool_registry=self._tool_registry,
            audio_processor=self._audio_processor
        )
        
        # Meeting simulator depends on io_pool and audio processor
        self._meeting_simulator = MeetingSimulator(
            io_pool=self.shared.io_pool,
            audio_processor=self._audio_processor
        )
        
        self._task_manager = TaskManager(
            audio_processor=self._audio_processor,
            vision_processor=self._vision_processor,
            llm_processor=self._llm_processor,
            tool_registry=self._tool_registry
        )
        
        # Initialize processors
        await self._audio_processor.initialize()
    
    async def cleanup(self):
        """Clean up all dependencies"""
        if self._task_manager:
            await self._task_manager.stop_all_tasks()
        
        await self.shared.cleanup()
    
    @property
    def audio_processor(self):
        return self._audio_processor
    
    @property
    def vision_processor(self):
        return self._vision_processor
    
    @property
    def video_recorder(self):
        return self._video_recorder
    
    @property
    def tool_registry(self):
        return self._tool_registry
    
    @property
    def llm_processor(self):
        return self._llm_processor
    
    @property
    def task_manager(self):
        return self._task_manager
    
    @property
    def meeting_simulator(self):
        return self._meeting_simulator


# Global container instance
container = DependencyContainer() 