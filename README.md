# Subject Matter Expert Voice Assistant

A real-time voice assistant that acts as a Subject Matter Expert (SME) in meetings, capable of joining virtual meetings, performing speaker diarization, transcribing conversations, and responding to queries using pre-loaded knowledge and meeting context.

## Features

- **Multi-Platform Meeting Support**: Join Zoom, Google Meet, and Microsoft Teams meetings
- **Real-Time Speaker Diarization**: Identify and separate up to 10 concurrent speakers
- **Streaming Transcription**: Real-time transcription with speaker labels
- **Wake Word Detection**: Configurable wake phrases (e.g., "Hey Expert")
- **Expert Responses**: RAG-based responses using domain knowledge and meeting context
- **Ultra-Low Latency**: Sub-500ms end-to-end response time
- **Full-Duplex Audio**: Natural interruption handling with barge-in support
- **Local & Cloud LLM**: Support for OpenAI, Gemini, and local Ollama models

## Installation

### Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- Microphone or virtual audio cable for meeting capture
- API keys for OpenAI/Gemini (or local Ollama installation)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Voice_SME_expert
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python setup.py install
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Create required directories:
```bash
mkdir -p knowledge/documents knowledge/embeddings knowledge/index
mkdir -p models/wake_word models/speaker_embedding
mkdir -p outputs logs
```

5. Index your knowledge base:
```bash
python scripts/index_knowledge.py --documents ./knowledge/documents --output ./knowledge/index
```

## Usage

### In-Person Meeting (Microphone)
```bash
python -m src.main --mode microphone --device default
```

### Virtual Meeting (Zoom)
```bash
python -m src.main --mode zoom --meeting-id 123456789 --password abc123 --name "AI Expert"
```

### Google Meet
```bash
python -m src.main --mode meet --meeting-url "https://meet.google.com/xxx-xxxx-xxx" --name "AI Expert"
```

### Microsoft Teams
```bash
python -m src.main --mode teams --meeting-url "https://teams.microsoft.com/..." --name "AI Expert"
```

### Virtual Audio Capture
```bash
python -m src.main --mode virtual --application zoom
```

## Configuration

Key configuration options in `.env`:

- **LLM Providers**: Configure OpenAI, Gemini, or Ollama
- **Wake Phrases**: Customize wake words
- **Audio Settings**: Sample rate, chunk size, device selection
- **Speaker Diarization**: Max speakers, similarity threshold
- **Knowledge Base**: Document paths and embedding configuration

## Project Structure

```
voice_expert_assistant/
├── config/              # Configuration modules
├── src/
│   ├── audio/          # Audio capture and processing
│   ├── diarization/    # Speaker diarization
│   ├── transcription/  # Speech-to-text
│   ├── wake_word/      # Wake word detection
│   ├── expert/         # Expert agent and RAG
│   ├── llm/            # LLM provider integrations
│   ├── tts/            # Text-to-speech
│   ├── meeting/        # Meeting platform connectors
│   └── utils/          # Utilities and helpers
├── knowledge/          # Knowledge base documents
├── models/             # ML models
├── scripts/            # Utility scripts
└── tests/              # Test suite
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Test specific components:
```bash
pytest tests/test_audio_capture.py -v
pytest tests/test_diarization.py -v
pytest tests/test_integration.py -v
```

## Scripts

### Test Audio Device
```bash
python scripts/test_audio_device.py --list
python scripts/test_audio_device.py --device 0 --test
```

### Enroll Speaker
```bash
python scripts/enroll_speaker.py --name "John Smith" --audio ./recordings/john_sample.wav
```

### Train Custom Wake Word
```bash
python scripts/train_wake_word.py --phrase "Hey Expert" --samples ./wake_word_samples/
```

## Performance Metrics

Expected latencies:
- STT: ~145ms average
- Speaker Diarization: ~89ms average
- LLM Response: ~890ms average
- TTS First Byte: ~52ms average
- End-to-End: ~1180ms average

## Hardware Requirements

### Minimum
- CPU: 4 cores (Intel i5 / AMD Ryzen 5)
- RAM: 8 GB
- Storage: 10 GB free space
- Network: Stable internet for cloud LLM

### Recommended
- CPU: 8 cores (Intel i7 / AMD Ryzen 7)
- RAM: 16 GB
- GPU: NVIDIA GPU with 4GB VRAM (for local models)
- Storage: 20 GB SSD
- Microphone: Professional USB microphone

## Troubleshooting

### Audio Issues
- Run `python scripts/test_audio_device.py --list` to check available devices
- Verify audio permissions in system settings
- For virtual meetings, ensure virtual audio cable is installed

### Meeting Connection
- Check meeting URLs and credentials
- Verify platform-specific SDK configuration
- Test with simple meeting join first

### Performance
- Reduce `MAX_SPEAKERS` if experiencing high latency
- Use local Ollama for offline operation
- Enable GPU support for faster processing

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Support

For issues and questions, please open an issue on GitHub.
