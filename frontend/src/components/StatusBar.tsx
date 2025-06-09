'use client';

import { Event, SystemStatus } from '@/lib/useSocket';
import { useState } from 'react';

interface StatusBarProps {
  isConnected: boolean;
  systemStatus: SystemStatus | null;
  lastEvent: Event | null;
  error: string | null;
  className?: string;
  refreshSystemStatus?: () => void;
}

export function StatusBar({ 
  isConnected, 
  systemStatus, 
  lastEvent, 
  error, 
  className = '',
  refreshSystemStatus 
}: StatusBarProps) {
  const [isToggling, setIsToggling] = useState(false);
  const [isCameraToggling, setIsCameraToggling] = useState(false);
  const [isAudioToggling, setIsAudioToggling] = useState(false);

  const getStatusColor = (status: boolean) => {
    return status ? 'text-green-400' : 'text-gray-500';
  };
  
  const getStatusClasses = (status: boolean) => {
    return status 
      ? 'status-indicator success' 
      : 'status-indicator text-gray-500';
  };

  const toggleVoiceDictation = async () => {
    if (!systemStatus || isToggling) return;
    
    setIsToggling(true);
    try {
      const response = await fetch('/api/voice/toggle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          enabled: !systemStatus.voice_dictation_enabled
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to toggle voice dictation');
      }
      
      const result = await response.json();
      
      // Refresh system status to get updated state
      if (refreshSystemStatus) {
        await refreshSystemStatus();
      }
      
    } catch (error) {
      console.error('Error toggling voice dictation:', error);
      // Optionally show user-visible error message
    } finally {
      setIsToggling(false);
    }
  };

  const toggleCameraCapture = async () => {
    if (!systemStatus || isCameraToggling) return;
    
    setIsCameraToggling(true);
    try {
      const response = await fetch('/api/camera/toggle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          enabled: !systemStatus.camera_capture_enabled
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to toggle camera capture');
      }
      
      const result = await response.json();
      
      // Refresh system status to get updated state
      if (refreshSystemStatus) {
        await refreshSystemStatus();
      }
      
    } catch (error) {
      console.error('Error toggling camera capture:', error);
      // Optionally show user-visible error message
    } finally {
      setIsCameraToggling(false);
    }
  };

  const toggleAudioInput = async () => {
    if (!systemStatus || isAudioToggling) return;
    
    setIsAudioToggling(true);
    try {
      const response = await fetch('/api/audio/toggle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          enabled: !systemStatus.audio_input_enabled
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to toggle audio input');
      }
      
      const result = await response.json();
      
      // Refresh system status to get updated state
      if (refreshSystemStatus) {
        await refreshSystemStatus();
      }
      
    } catch (error) {
      console.error('Error toggling audio input:', error);
      // Optionally show user-visible error message
    } finally {
      setIsAudioToggling(false);
    }
  };
  
  const getLastEventDisplay = () => {
    if (!lastEvent) return { icon: '📡', text: 'No events', color: 'text-gray-400' };
    
    // Handle new consolidated event system: type:action
    const eventKey = `${lastEvent.type}:${lastEvent.action}`;
    
    const eventDisplayMap: Record<string, { icon: string; text: string; color: string }> = {
      // System status events
      'system_status:listening': { icon: '🎤', text: 'Audio listening started', color: 'text-blue-400' },
      'system_status:camera_active': { icon: '📹', text: 'Camera activated', color: 'text-green-400' },
      'system_status:whisper_loading': { icon: '⏳', text: 'Loading Whisper model', color: 'text-yellow-400' },
      'system_status:whisper_ready': { icon: '✅', text: 'Whisper model ready', color: 'text-green-400' },
      'system_status:system_ready': { icon: '🚀', text: 'System ready', color: 'text-green-400' },
      
      // Audio events
      'audio_event:transcription_start': { icon: '🎯', text: 'Starting transcription', color: 'text-blue-400' },
      'audio_event:transcription_end': { icon: '✅', text: 'Transcription complete', color: 'text-green-400' },
      'audio_event:raw_transcript': { icon: '📝', text: 'Raw transcript received', color: 'text-gray-400' },
      'audio_event:wake_word_detected': { icon: '🎤', text: 'Wake word detected', color: 'text-orange-400' },
      'audio_event:context_ready': { icon: '📋', text: 'Context ready for AI', color: 'text-purple-400' },
      'audio_event:audio_input_toggled': { 
        icon: lastEvent?.data?.enabled ? '🎤' : '🔇', 
        text: `Audio input ${lastEvent?.data?.enabled ? 'enabled' : 'disabled'}`, 
        color: lastEvent?.data?.enabled ? 'text-green-400' : 'text-orange-400' 
      },
      'audio_event:voice_dictation_toggled': { 
        icon: lastEvent?.data?.enabled ? '🔊' : '🔇', 
        text: `Voice dictation ${lastEvent?.data?.enabled ? 'enabled' : 'disabled'}`, 
        color: lastEvent?.data?.enabled ? 'text-green-400' : 'text-orange-400' 
      },
      
      // LLM events
      'llm_event:response_start': { icon: '🤖', text: 'AI thinking', color: 'text-cyan-400' },
      'llm_event:response_chunk': { icon: '💭', text: 'AI responding', color: 'text-cyan-400' },
      'llm_event:response_end': { icon: '✅', text: 'Response complete', color: 'text-green-400' },
      
      // TTS events
      'tts_event:start': { icon: '🔊', text: 'Speaking', color: 'text-indigo-400' },
      'tts_event:end': { icon: '🔇', text: 'Speech complete', color: 'text-gray-400' },
      
      // Voice control events
      'voice_control:dictation_toggled': { 
        icon: lastEvent?.data?.enabled ? '🎤' : '🔇', 
        text: `Voice dictation ${lastEvent?.data?.enabled ? 'enabled' : 'disabled'}`, 
        color: lastEvent?.data?.enabled ? 'text-green-400' : 'text-orange-400' 
      },
      
      // Camera control events
      'camera_control:capture_toggled': { 
        icon: lastEvent?.data?.enabled ? '📸' : '🔇', 
        text: `Camera capture ${lastEvent?.data?.enabled ? 'enabled' : 'disabled'}`, 
        color: lastEvent?.data?.enabled ? 'text-green-400' : 'text-orange-400' 
      },
      
      // Vision events
      'vision_event:frame_captured': { icon: '📷', text: 'Frame captured', color: 'text-teal-400' },
      
      // Error events
      'error:camera_init_failed': { icon: '❌', text: 'Camera initialization failed', color: 'text-red-400' },
      'error:whisper_failed': { icon: '❌', text: 'Whisper model failed', color: 'text-red-400' },
      'error:llm_processing_failed': { icon: '❌', text: 'AI processing failed', color: 'text-red-400' },
      'error:tts_failed': { icon: '❌', text: 'Text-to-speech failed', color: 'text-red-400' },
      'error:audio_loop_error': { icon: '❌', text: 'Audio processing error', color: 'text-red-400' },
    };
    
    const eventInfo = eventDisplayMap[eventKey];
    return eventInfo || { icon: '📡', text: `Event: ${lastEvent.type}:${lastEvent.action}`, color: 'text-gray-400' };
  };
  
  const eventInfo = getLastEventDisplay();
  
  return (
    <div className={`elevated-card rounded-xl flex flex-col overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-gray-800">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center text-xs shadow-lg">
            📊
          </div>
          <h2 className="text-base font-semibold text-white">System Status</h2>
        </div>
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
      
        {/* Connection Status */}
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <h3 className="text-xs font-medium text-gray-300 uppercase tracking-wide">Connection</h3>
              <div className="flex items-center">
                <div className={getStatusClasses(isConnected)}>
                  <span className="text-sm font-medium">
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>
              {error && (
                <div className="mt-2 p-2 bg-red-900/30 border border-red-800/50 rounded-lg">
                  <p className="text-xs text-red-300 leading-relaxed">
                    {error}
                  </p>
                </div>
              )}
            </div>
            
            <div className="space-y-2">
              <h3 className="text-xs font-medium text-gray-300 uppercase tracking-wide">System</h3>
              <div className="flex items-center">
                <div className={getStatusClasses(systemStatus?.is_running || false)}>
                  <span className="text-sm font-medium">
                    {systemStatus?.is_running ? 'Running' : 'Stopped'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      
        {/* Component Status */}
        {systemStatus && (
          <div className="space-y-3">
            <h3 className="text-xs font-medium text-gray-300 uppercase tracking-wide">Components</h3>
            <div className="grid grid-cols-1 gap-2">
              <div className="flex items-center justify-between p-2 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <span className="text-xs text-gray-300 font-medium">Audio Listening</span>
                <div className={getStatusClasses(systemStatus.audio_listening)}>
                  <span className="text-xs font-medium">
                    {systemStatus.audio_listening ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-2 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <span className="text-xs text-gray-300 font-medium">Audio Input</span>
                <div className="flex items-center space-x-2">
                  <div className={getStatusClasses(systemStatus.audio_input_enabled)}>
                    <span className="text-xs font-medium">
                      {systemStatus.audio_input_enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                  <button
                    onClick={toggleAudioInput}
                    disabled={!isConnected || isAudioToggling}
                    className={`
                      px-2 py-1 text-xs rounded-md font-medium transition-colors
                      ${systemStatus.audio_input_enabled 
                        ? 'bg-orange-600 hover:bg-orange-500 text-white' 
                        : 'bg-green-600 hover:bg-green-500 text-white'
                      }
                      ${(!isConnected || isAudioToggling) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                    `}
                  >
                    {isAudioToggling ? '...' : (systemStatus.audio_input_enabled ? 'Disable' : 'Enable')}
                  </button>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-2 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <span className="text-xs text-gray-300 font-medium">Voice Dictation</span>
                <div className="flex items-center space-x-2">
                  <div className={getStatusClasses(systemStatus.voice_dictation_enabled)}>
                    <span className="text-xs font-medium">
                      {systemStatus.voice_dictation_enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                  <button
                    onClick={toggleVoiceDictation}
                    disabled={!isConnected || isToggling}
                    className={`
                      px-2 py-1 text-xs rounded-md font-medium transition-colors
                      ${systemStatus.voice_dictation_enabled 
                        ? 'bg-orange-600 hover:bg-orange-500 text-white' 
                        : 'bg-green-600 hover:bg-green-500 text-white'
                      }
                      ${(!isConnected || isToggling) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                    `}
                  >
                    {isToggling ? '...' : (systemStatus.voice_dictation_enabled ? 'Disable' : 'Enable')}
                  </button>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-2 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <span className="text-xs text-gray-300 font-medium">Camera Capture</span>
                <div className="flex items-center space-x-2">
                  <div className={getStatusClasses(systemStatus.camera_capture_enabled)}>
                    <span className="text-xs font-medium">
                      {systemStatus.camera_capture_enabled ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <button
                    onClick={toggleCameraCapture}
                    disabled={!isConnected || isCameraToggling}
                    className={`
                      px-2 py-1 text-xs rounded-md font-medium transition-colors
                      ${systemStatus.camera_capture_enabled 
                        ? 'bg-orange-600 hover:bg-orange-500 text-white' 
                        : 'bg-green-600 hover:bg-green-500 text-white'
                      }
                      ${(!isConnected || isCameraToggling) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                    `}
                  >
                    {isCameraToggling ? '...' : (systemStatus.camera_capture_enabled ? 'Disable' : 'Enable')}
                  </button>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-2 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <span className="text-xs text-gray-300 font-medium">AI Processing</span>
                <div className={getStatusClasses(systemStatus.llm_processing)}>
                  <span className="text-xs font-medium">
                    {systemStatus.llm_processing ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-2 bg-gray-800/50 rounded-lg border border-gray-700/50">
                <span className="text-xs text-gray-300 font-medium">Whisper Model</span>
                <div className={getStatusClasses(systemStatus.whisper_loaded)}>
                  <span className="text-xs font-medium">
                    {systemStatus.whisper_loaded ? 'Loaded' : 'Not Loaded'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Latest Activity */}
        <div className="space-y-3">
          <h3 className="text-xs font-medium text-gray-300 uppercase tracking-wide">Latest Activity</h3>
          <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-3">
            <div className="flex items-start space-x-2">
              <span className="text-base flex-shrink-0">{eventInfo.icon}</span>
              <div className="flex-1 min-w-0">
                <p className={`text-xs font-medium ${eventInfo.color}`}>
                  {eventInfo.text}
                </p>
                {lastEvent && (
                  <p className="text-xs text-gray-500 mt-1">
                    {new Date(lastEvent.timestamp).toLocaleTimeString()}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 