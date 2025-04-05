# AI Call Assistant

An intelligent voice assistant that can analyze speech, detect potential fraud, and engage in natural conversations with users.

## Overview

The AI Call Assistant is a sophisticated tool that combines speech recognition, natural language processing, and fraud detection capabilities. It processes spoken input, transcribes it to text, analyzes the content for fraudulent patterns, and generates appropriate responses. The assistant can also extract useful information from conversations, such as names, contact details, and the purpose of the call.

## Features

- **Real-time Voice Processing**: Record and process audio in real-time
- **Speech-to-Text**: Convert spoken words to text using advanced transcription models
- **Fraud Detection**: Analyze conversation content to detect potential scams or fraudulent activity
- **Intelligent Responses**: Generate contextually appropriate responses based on the conversation
- **Text-to-Speech**: Convert responses to natural-sounding speech
- **Information Extraction**: Automatically extract useful data points like names and contact information
- **Conversation State Management**: Track the state of the conversation to provide coherent interactions
- **User-friendly Interface**: Clean, intuitive web interface for interacting with the assistant

## Technical Architecture

The AI Call Assistant consists of several key components:

1. **Frontend**: HTML interface for recording audio and displaying results
2. **API Server**: Flask-based backend that handles audio processing and response generation
3. **Speech Processing**: Whisper-based transcription engine for speech-to-text conversion
4. **Classification Model**: ML model trained to detect fraudulent patterns in conversations
5. **Response Generator**: Component that creates appropriate conversational responses
6. **Text-to-Speech Engine**: Google TTS for converting text responses to audio

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Microphone for audio input
- Speakers for audio output

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/AI_Hackathon_IITM_bot.git
cd AI_Hackathon_IITM_bot
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download the required models:
```
python -c "import whisper; whisper.load_model('tiny')"
```

### Running the Application

1. Start the API server:
```
cd Code
python api.py
```

2. Open the web interface:
   - Open `Code/webapp.html` in your web browser
   - Or serve it using a simple HTTP server: `python -m http.server 8000`

3. Use the assistant:
   - Click the "Call" button to start recording
   - Speak into your microphone
   - Click the button again to stop recording
   - View the analysis and hear the assistant's response

## Usage Example

1. **Start a Call**: Click the "Call" button and speak a query
2. **View Analysis**: The system will display:
   - Transcription of your speech
   - Classification of the content
   - Probability scores for different categories
   - Any extracted information
3. **Hear Response**: The AI assistant will respond both textually and with synthesized speech
4. **Continue Conversation**: Click "Call" again to continue the conversation

## Development

### Project Structure

```
AI_Hackathon_IITM_bot/
│
├── Code/
│   ├── api.py                    # API server with endpoints for audio processing
│   ├── analyze_audio.py          # Audio analysis and classification functions
│   ├── generative_response_bot.py # Response generation logic
│   └── webapp.html               # Web interface for the assistant
│
├── Models/
│   └── voice_text_classifier.joblib # Pre-trained classification model
│
└── logs/                         # Directory for conversation logs
```

### Customization

- **Fraud Threshold**: Adjust `fraud_threshold` in the `ResponseGenerator` initialization
- **TTS Voice**: Modify the `gtts` parameters in `text_to_speech` function to change language or speed
- **Response Templates**: Edit the templates in `_get_default_templates` method of `ResponseGenerator`

