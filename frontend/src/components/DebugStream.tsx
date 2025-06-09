import { useState, useRef, useEffect } from 'react';
import { Event } from '@/lib/useSocket';

interface RawTranscript {
  id: string;
  content: string;
  timestamp: Date;
  hasWakeWord: boolean;
  isProcessed: boolean;
  isInContext: boolean;
}

interface DebugStreamProps {
  lastEvent: Event | null;
  className?: string;
}

export function DebugStream({ lastEvent, className = '' }: DebugStreamProps) {
  const [rawTranscripts, setRawTranscripts] = useState<RawTranscript[]>([]);
  const [showDebugPanel, setShowDebugPanel] = useState(true);
  
  const isTranscribing = useRef(false);
  const isInContextMode = useRef(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  
  // Check if transcript contains wake words
  const containsWakeWord = (transcript: string): boolean => {
    const wakeWords = ["osmo", "hey osmo", "hey"];
    const lower = transcript.toLowerCase();
    return wakeWords.some(word => lower.includes(word));
  };
  
  useEffect(() => {
    if (!lastEvent) return;
    
    const eventType = lastEvent.type;
    const eventAction = lastEvent.action;
    
    switch (eventType) {
      case 'audio_event':
        switch (eventAction) {
          case 'transcription_start':
            isTranscribing.current = true;
            break;
            
          case 'transcription_end':
            isTranscribing.current = false;
            break;
            
          case 'raw_transcript':
          case 'utterance_ready':
            const rawTranscript = lastEvent.data?.transcript as string;
            if (rawTranscript && typeof rawTranscript === 'string' && rawTranscript.trim()) {
              const hasWakeWord = containsWakeWord(rawTranscript);
              const newRawTranscript: RawTranscript = {
                id: Date.now().toString(),
                content: rawTranscript,
                timestamp: new Date(),
                hasWakeWord,
                isProcessed: false,
                isInContext: isInContextMode.current && !hasWakeWord
              };
              setRawTranscripts(prev => {
                const updated = [...prev, newRawTranscript];
                // Keep only last 50 transcripts (increased for continuous mode)
                return updated.slice(-50);
              });
            }
            break;
            
          case 'wake_word_detected':
            // Start context accumulation mode
            isInContextMode.current = true;
            
            // Mark the most recent transcript as having triggered wake word detection
            setRawTranscripts(prev => 
              prev.map((transcript, index) => 
                index === prev.length - 1 ? { ...transcript, isProcessed: true } : transcript
              )
            );
            break;
            
          case 'context_ready':
            // Context accumulation is now complete
            isInContextMode.current = false;
            isTranscribing.current = false;
            break;
            
          case 'continuous_mode_enabled':
            // Clear existing transcripts when entering continuous mode
            setRawTranscripts([]);
            break;
        }
        break;
    }
  }, [lastEvent]);
  
  // Auto-scroll to bottom when new transcripts are added
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [rawTranscripts]);
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  
  if (!showDebugPanel) {
    return (
      <div className={`elevated-card rounded-xl p-5 h-full flex flex-col justify-center ${className}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3 min-w-0 flex-1">
            <div className="w-8 h-8 bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-lg flex items-center justify-center text-sm shadow-lg">
              🔍
            </div>
            <h2 className="text-lg font-semibold text-white">Debug Stream</h2>
          </div>
          <button
            onClick={() => setShowDebugPanel(true)}
            className="text-xs bg-gray-700/50 hover:bg-gray-600/50 text-gray-300 px-3 py-1.5 rounded-lg transition-all duration-200 border border-gray-600/50"
          >
            Show
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div className={`elevated-card rounded-xl flex flex-col h-full overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 flex-shrink-0">
        <div className="flex items-center space-x-2 min-w-0 flex-1">
          <div className="w-6 h-6 bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-lg flex items-center justify-center text-xs shadow-lg">
            🔍
          </div>
          <h2 className="text-base font-semibold text-white">Debug Stream</h2>
          <div className="flex items-center space-x-2">
            {isTranscribing.current && (
              <div className="status-indicator info">
                <span className="text-xs font-medium">Transcribing</span>
              </div>
            )}
            {isInContextMode.current && (
              <div className="status-indicator warning">
                <span className="text-xs font-medium">Context Mode</span>
              </div>
            )}
          </div>
        </div>
        <button
          onClick={() => setShowDebugPanel(false)}
          className="text-xs bg-gray-700/50 hover:bg-gray-600/50 text-gray-300 px-2 py-1 rounded-lg transition-all duration-200 border border-gray-600/50"
        >
          Hide
        </button>
      </div>
      
      {/* Content */}
      <div 
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-2 text-sm min-h-0 max-h-full"
      >
        {rawTranscripts.length === 0 && (
          <div className="text-center py-6">
            <div className="w-10 h-10 bg-gradient-to-br from-gray-700 to-gray-800 rounded-xl flex items-center justify-center text-base mx-auto mb-3 opacity-60">
              👂
            </div>
            <p className="text-gray-400 font-medium text-sm">Listening for speech...</p>
          </div>
        )}
        
        {rawTranscripts.map((transcript) => (
          <div
            key={transcript.id}
            className={`p-3 rounded-lg border transition-all duration-200 fade-in ${
              transcript.hasWakeWord
                ? transcript.isProcessed
                  ? 'bg-green-900/30 border-green-700/50 text-green-200'
                  : 'bg-yellow-900/30 border-yellow-700/50 text-yellow-200'
                : transcript.isInContext
                  ? 'bg-blue-900/30 border-blue-700/50 text-blue-200'
                  : 'bg-gray-800/50 border-gray-700/50 text-gray-300'
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="flex-1 break-words min-w-0 leading-relaxed font-mono text-xs">
                {transcript.content}
              </span>
              <div className="flex items-center space-x-1 flex-shrink-0">
                {transcript.hasWakeWord && (
                  <span className={`px-1.5 py-0.5 rounded text-xs font-medium transition-all ${
                    transcript.isProcessed 
                      ? 'bg-green-600/80 text-green-100 border border-green-500/50' 
                      : 'bg-yellow-600/80 text-yellow-100 border border-yellow-500/50'
                  }`}>
                    {transcript.isProcessed ? '→ AI' : 'WAKE'}
                  </span>
                )}
                {transcript.isInContext && (
                  <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-blue-600/80 text-blue-100 border border-blue-500/50">
                    CTX
                  </span>
                )}
                <span className="text-gray-500 text-xs font-mono">
                  {formatTime(transcript.timestamp)}
                </span>
              </div>
            </div>
          </div>
        ))}

      </div>
    </div>
  );
} 