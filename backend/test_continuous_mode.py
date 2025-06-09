#!/usr/bin/env python3
"""
Test script for continuous mode functionality
This script tests the continuous audio processing without wake words
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.core.shared import container
from backend.events import event_bus, Event, EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_continuous_mode():
    """Test continuous mode functionality"""
    print("🎯 Testing Continuous Mode Functionality")
    print("=" * 50)
    
    try:
        # Initialize container
        print("📋 Initializing system...")
        await container.initialize()
        
        # Subscribe to events
        event_queue = await event_bus.subscribe()
        
        # Test 1: Enable continuous mode
        print("\n1️⃣ Testing continuous mode activation...")
        await container.audio_processor.enable_continuous_mode()
        
        # Verify continuous mode is enabled
        if container.audio_processor.continuous_mode:
            print("✅ Continuous mode enabled successfully")
        else:
            print("❌ Failed to enable continuous mode")
            return
        
        # Test 2: Start audio processing
        print("\n2️⃣ Starting audio processing...")
        await container.audio_processor.start_listening()
        
        # Test 3: Wait for events and monitor
        print("\n3️⃣ Monitoring for utterance events...")
        print("💬 Speak into your microphone - all speech will be processed without wake words")
        print("🔇 Utterances will be detected after 500ms of silence")
        print("⏹️  Press Ctrl+C to stop\n")
        
        utterance_count = 0
        
        try:
            while True:
                try:
                    # Wait for events with timeout
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    
                    if event.type == EventType.AUDIO_EVENT:
                        if event.action == "continuous_mode_enabled":
                            print(f"🎧 {event.data.get('message', 'Continuous mode activated')}")
                        
                        elif event.action == "utterance_ready":
                            utterance_count += 1
                            transcript = event.data.get('transcript', '')
                            timestamp = event.data.get('timestamp', 0)
                            
                            print(f"🎤 Utterance #{utterance_count}: '{transcript}'")
                            print(f"   Timestamp: {timestamp}")
                            
                        elif event.action == "speech_start":
                            print("🗣️  Speech detected...")
                            
                        elif event.action == "speech_end":
                            print("🤫 Silence detected - processing utterance...")
                    
                    elif event.type == EventType.SYSTEM_STATUS:
                        print(f"📊 System: {event.action} - {event.data}")
                        
                except asyncio.TimeoutError:
                    # No events in the last second, just continue
                    pass
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping test...")
        
        # Test 4: Disable continuous mode
        print("\n4️⃣ Disabling continuous mode...")
        await container.audio_processor.disable_continuous_mode()
        
        if not container.audio_processor.continuous_mode:
            print("✅ Continuous mode disabled successfully")
        else:
            print("❌ Failed to disable continuous mode")
        
        # Stop audio processing
        await container.audio_processor.stop_listening()
        
        # Summary
        print(f"\n📈 Test Summary:")
        print(f"   - Utterances processed: {utterance_count}")
        print(f"   - Continuous mode: {'✅ Working' if utterance_count > 0 else '⚠️  No speech detected'}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    
    finally:
        await container.cleanup()
    
    print("\n🎉 Continuous mode test completed!")
    return True


async def test_meeting_simulator():
    """Test meeting simulator integration"""
    print("\n🎬 Testing Meeting Simulator Integration")
    print("=" * 50)
    
    try:
        # Test meeting simulator availability
        meeting_sim = container.meeting_simulator
        if meeting_sim:
            print("✅ Meeting simulator available")
            
            # Test status
            status = meeting_sim.get_current_status()
            print(f"📊 Meeting status: {status['status']}")
            
            # Test continuous mode integration
            if meeting_sim.audio_processor:
                print("✅ Audio processor integrated")
            else:
                print("❌ Audio processor not integrated")
                
        else:
            print("❌ Meeting simulator not available")
            
    except Exception as e:
        logger.error(f"Meeting simulator test failed: {e}")


if __name__ == "__main__":
    async def main():
        await test_continuous_mode()
        await test_meeting_simulator()
    
    asyncio.run(main()) 