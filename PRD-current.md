# VoiceGraph - Real-time Conversation Intelligence System
## Product Requirements Document (PRD)

---

## **1. Executive Summary**

**Project**: VoiceGraph  
**Vision**: Transform conversations into actionable intelligence  
**Mission**: Provide real-time transcription, speaker identification, topic analysis, and AI-powered suggestions during meetings and conversations

VoiceGraph removes the friction of traditional meeting analysis by processing audio continuously, identifying speakers automatically, and generating contextual insights as conversations unfold.

---

## **2. Product Overview**

### **2.1 Current Implementation Phase**
**Phase**: Meeting Simulation with Continuous Processing  
**Status**: Implementing core audio pipeline transformation  
**Sprint Goal**: Remove wake word dependency and enable continuous conversation analysis

### **2.2 Core Value Proposition**
- **Eliminate Manual Note-taking**: Automatic transcription with speaker identification
- **Real-time Intelligence**: Topics, action items, and suggestions as they emerge
- **Conversation Health**: Monitor discussion flow and engagement patterns
- **Meeting Optimization**: AI-powered suggestions to improve conversation quality

---

## **3. Technical Architecture**

### **3.1 Audio Processing Pipeline**
```
Microphone → VAD → Utterance Detection → MLX Whisper → Transcript
                           ↓                              ↓
                   Speaker Diarization              Topic Extraction
                       (Pyannote)                        ↓
                           ↓                      AI Analysis Engine
                   Speaker Assignment                     ↓
                           ↓                    Real-time Suggestions
                    Timestamped Segments              & Insights
```

**Key Components**:
- **Continuous Mode**: Process all speech without wake words
- **Utterance Detection**: 500ms silence threshold for natural boundaries
- **Speaker Diarization**: Pyannote.audio 3.1 integration
- **Transcription**: MLX Whisper (local, fast, private)

### **3.2 Analysis Pipeline**
```
Utterance → Transcription → Topic Extraction → AI Analysis → UI Update
           ↓                ↓                   ↓
     Speaker ID        Concept Detection   Suggestions Generation
```

### **3.3 LLM Integration Strategy**
**Primary Models**:
- **DeepSeek V3**: Cost-effective, good quality ($0.001/1K tokens)
- **GPT-4o**: High quality baseline ($0.03/1K tokens)
- **Llama 3.1 70B**: Open source option via Cerebras ($0.006/1K tokens)

**Model Selection Criteria**:
- Response quality and relevance
- Latency (target: <3s)
- Cost optimization
- A/B testing capabilities

---

## **4. Core Features (Current Sprint)**

### **4.1 Audio Pipeline Transformation** 🔥 **[PRIORITY]**
| Feature | Status | Description | Success Criteria |
|---------|--------|-------------|------------------|
| Remove Wake Words | ❌ | Eliminate "Osmo" activation requirement | All speech processed continuously |
| Continuous Mode | ❌ | Enable utterance boundary detection | 500ms silence threshold accuracy |
| Live Processing | ❌ | Test with microphone input | <2s transcript latency |
| Event System | ❌ | Update for utterance_ready events | Seamless event flow |

### **4.2 Meeting Simulation System**
| Feature | Status | Description | Success Criteria |
|---------|--------|-------------|------------------|
| File Upload | ❌ | Drag-drop audio upload | Support .wav, .mp3, .m4a |
| Speaker Diarization | ❌ | Pyannote.audio integration | 7-25% DER accuracy |
| Real-time Playback | ❌ | Simulate meeting timing | Variable speed (0.5x-3x) |
| Synchronized Transcript | ❌ | Speaker-labeled text | Real-time display |

### **4.3 AI Analysis Engine**
| Feature | Status | Description | Success Criteria |
|---------|--------|-------------|------------------|
| Topic Extraction | ❌ | Current topics identification | >80% relevance |
| Action Items | ❌ | Automatic commitment detection | Clear, actionable items |
| Suggestions | ❌ | Context-aware conversation prompts | Timely, helpful insights |
| Research Triggers | ❌ | Knowledge gap identification | Relevant search suggestions |

### **4.4 User Interface**
| Component | Status | Description | Success Criteria |
|-----------|--------|-------------|------------------|
| Meeting Dashboard | ❌ | 4-panel layout | Intuitive, responsive |
| Live Transcript | ❌ | Real-time text stream | Speaker colors, timestamps |
| Analysis Panel | ❌ | Topics, entities, sentiment | Dynamic updates |
| Playback Controls | ❌ | Play/pause/speed controls | Smooth operation |

---

## **5. Success Metrics**

### **5.1 Performance Targets (Current Sprint)**
- **Utterance Detection Accuracy**: >95%
- **Transcription Latency**: <2s from speech end
- **Topic Extraction Relevance**: >80% user satisfaction
- **Model Response Time**: <3s for insights
- **Upload Processing**: <30s for 5min audio
- **UI Responsiveness**: <50ms update latency

### **5.2 Quality Metrics**
- **Diarization Accuracy**: 7-25% DER (pyannote benchmark)
- **Transcription Accuracy**: >95% WER for clear speech
- **Suggestion Relevance**: >70% user approval
- **System Reliability**: >99% uptime during testing

### **5.3 Cost Optimization**
- **Target Cost**: <$0.01 per conversation minute
- **Model Efficiency**: Optimal model selection per use case
- **Local Processing**: Minimize cloud API usage

---

## **6. User Journey & Use Cases**

### **6.1 Primary Use Case: Meeting Analysis Testing**
1. **Upload**: User drags meeting audio file
2. **Processing**: System performs diarization (~30s)
3. **Preview**: Display speaker count and timeline
4. **Simulate**: Start real-time playback
5. **Analyze**: Watch transcript and insights appear live
6. **Iterate**: Adjust speed, test different models

### **6.2 Future Use Case: Live Meeting Support**
1. **Join**: Start conversation analysis
2. **Listen**: Continuous processing of all speakers
3. **Transcribe**: Real-time transcript with speaker labels
4. **Analyze**: Topic tracking and suggestion generation
5. **Act**: Follow up on action items and insights

---

## **7. Technical Implementation Plan**

### **7.1 Phase 1: Audio Pipeline Transformation** (30 min)
```python
# Critical Path - Must complete first
- Remove wake word detection from audio.py
- Add continuous_mode flag and methods
- Update VAD processing for utterance boundaries
- Test with live microphone input
```

### **7.2 Phase 2: Backend Integration** (20 min)
```python
# Build on Phase 1
- Integrate MeetingSimulator into dependency container
- Add continuous mode activation
- Create meeting API endpoints
- Update event system for utterance_ready events
```

### **7.3 Phase 3: Frontend Development** (35 min)
```typescript
// User interface for testing
- Create /meeting route and dashboard
- Build file upload with progress
- Real-time transcript display
- Analysis panels (topics, suggestions)
- WebSocket event handling
```

### **7.4 Phase 4: Testing & Refinement** (15 min)
```bash
# Validation and polish
- Test continuous mode with live audio
- Verify meeting simulation accuracy
- Check AI analysis quality
- Polish UI responsiveness
```

**Total Implementation Time**: ~100 minutes

---

## **8. Risk Assessment & Mitigation**

### **8.1 Technical Risks**
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Wake word removal breaks system | Medium | High | Thorough testing, gradual rollout |
| Pyannote model access issues | Low | Medium | Mock diarization fallback |
| Audio processing latency | Medium | High | Local MLX Whisper, optimized pipeline |
| LLM API rate limits | Low | Medium | Multiple provider support |

### **8.2 Quality Risks**
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Poor diarization accuracy | Medium | Medium | High-quality test audio, tuning |
| Irrelevant AI suggestions | High | Medium | Prompt engineering, user feedback |
| Transcript accuracy issues | Low | High | Clear audio requirements, fallbacks |

---

## **9. Future Roadmap**

### **9.1 Phase 2: Multi-Agent Architecture** (Next Sprint)
```python
# Advanced conversation analysis
class ConversationAgents:
    - MeetingHealthAgent: Flow and engagement monitoring
    - TopicCoherenceAgent: Drift and focus detection  
    - ActionDetectionAgent: Commitment and task tracking
    - ResearchAgent: Knowledge gap identification
```

### **9.2 Phase 3: Graph Visualization** (Future)
- **D3.js Force-Directed Graph**: Real-time conversation mapping
- **Semantic Clustering**: Topic and concept relationships
- **Speaker Networks**: Interaction patterns and engagement
- **Timeline Visualization**: Conversation evolution over time

### **9.3 Phase 4: Production Features**
- **Live Microphone Support**: Real-time meeting participation
- **Multi-User Sessions**: UUID-based room system
- **Export Capabilities**: Transcript, summary, action items
- **Integration APIs**: Slack, Teams, Zoom integration

---

## **10. Competitive Analysis**

### **10.1 Current Solutions**
- **Otter.ai**: Good transcription, limited real-time analysis
- **Rev.ai**: High accuracy, no conversation intelligence
- **Krisp**: Noise cancellation, no content analysis
- **Grain**: Recording focus, post-meeting analysis

### **10.2 VoiceGraph Advantages**
- **Real-time Processing**: Insights during conversation, not after
- **Local Privacy**: Audio processed locally, only transcripts to cloud
- **Continuous Mode**: No wake words or manual activation
- **Multi-Model**: A/B testing different LLMs for optimal results
- **Open Architecture**: Extensible agent system

---

## **11. Data & Privacy**

### **11.1 Privacy-First Design**
- **Local Audio Processing**: MLX Whisper runs locally
- **Transcript-Only Cloud**: Only text sent to LLM APIs
- **No Audio Storage**: Raw audio discarded after processing
- **User Control**: Clear data handling transparency

### **11.2 Data Flow**
```
Audio Input → Local Transcription → Cloud Analysis → Local Storage
    ↓              ↓                    ↓              ↓
 Deleted      Text Only         Insights Only    User Controlled
```

---

## **12. Success Definition**

### **12.1 Current Sprint Success**
- ✅ **Continuous processing** works without wake words
- ✅ **Meeting simulation** demonstrates real-time analysis
- ✅ **AI insights** provide relevant, timely suggestions
- ✅ **User experience** feels natural and responsive
- ✅ **Technical foundation** supports future features

### **12.2 Long-term Vision Success**
- **Adoption**: 1000+ users testing meeting analysis
- **Quality**: >90% user satisfaction with insights
- **Performance**: <$0.005 per conversation minute
- **Scale**: Support 10+ concurrent speakers for 2+ hours
- **Impact**: Measurably improved meeting outcomes

---

## **13. Resource Requirements**

### **13.1 Development Resources**
- **Current Sprint**: 1 developer, ~2 hours implementation
- **Testing**: Sample meeting audio files
- **Environment**: Hugging Face token, OpenRouter API key

### **13.2 Infrastructure Requirements**
- **Local**: 4GB+ RAM, macOS for MLX Whisper
- **Cloud**: OpenRouter API credits
- **Optional**: CUDA GPU for faster diarization

---

## **14. Acceptance Criteria**

### **14.1 Phase 1 Complete When:**
- [ ] Audio processor works without wake words
- [ ] Continuous mode processes all speech
- [ ] Utterance boundaries detected accurately
- [ ] Live microphone testing successful

### **14.2 Meeting Simulation Complete When:**
- [ ] Audio files upload successfully
- [ ] Pyannote diarization shows speaker timeline
- [ ] Real-time playback with synchronized transcript
- [ ] AI analysis generates relevant insights
- [ ] UI updates smoothly in real-time

### **14.3 Production Ready When:**
- [ ] >95% utterance detection accuracy
- [ ] <2s transcript latency
- [ ] >80% AI suggestion relevance
- [ ] System handles 5+ minute meetings smoothly
- [ ] Error handling and recovery mechanisms

---

**This PRD serves as the blueprint for transforming Osmo Assistant into VoiceGraph, prioritizing continuous processing as the foundation for all advanced conversation intelligence features.**
