# Voice Analyzer

A web-based application that records audio, transcribes speech, and classifies the content using machine learning.

## Overview

Voice Analyzer is an interactive web application that allows users to record their voice, then automatically transcribes and classifies the speech content. The system uses Whisper for speech-to-text transcription and a pre-trained machine learning model to classify the transcribed text into different categories.

## Features

- **Real-time Voice Recording**: Record audio directly from your browser
- **Automatic Transcription**: Convert speech to text using OpenAI's Whisper model
- **Text Classification**: Analyze the transcribed text and classify it into categories
- **Visual Results**: View classification results with probability scores as progress bars
- **Downloadable Recordings**: Save your recordings for later use

## System Architecture

The application consists of:

1. **Frontend**: A web interface (webapp.html) for recording audio and displaying results
2. **Backend API**: A Flask server (api.py) that handles audio processing
3. **Analysis Engine**: Python scripts for transcription and classification (analyze_audio.py)
4. **ML Model**: A pre-trained classifier for text categorization (voice_text_classifier.joblib)

## Setup and Installation

### Prerequisites

- Python 3.8+ 
- Flask and Flask-CORS
- OpenAI Whisper
- scikit-learn
- joblib
- Required audio processing libraries

### Installation Steps

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd AI_Hackathon
   ```

2. Install required Python packages:
   ```bash
   pip install flask flask-cors openai-whisper scikit-learn joblib torch numpy
   ```

3. Ensure the trained classifier model is available at the root directory:
   - The application expects a file named `voice_text_classifier.joblib`

## Running the Application

1. Start the API server:
   ```bash
   python api.py
   ```
   This will start the Flask server on http://localhost:5000

2. Open the webapp.html file in a web browser:
   - You can use any modern web browser (Chrome, Firefox, Edge, etc.)
   - For security reasons, some browsers may require serving the HTML through a web server rather than opening it directly as a file

## How to Use

1. **Recording Audio**:
   - Click the "Start Recording" button to begin recording
   - Speak into your microphone
   - Click "Stop Recording" when finished

2. **Viewing Results**:
   - After recording stops, the audio is automatically sent for analysis
   - A loading spinner indicates processing is in progress
   - Results will appear showing:
     - The transcribed text
     - The classification result
     - Probability scores for each category

3. **Saving Recordings**:
   - Each recording has a "Save" button
   - Click to download the audio file (.wav format) to your device

## Technical Details

### API Endpoints

The API server provides the following endpoint:

- **POST /analyze**: Analyzes audio content
  - Accepts multipart/form-data with an 'audio' file
  - Returns JSON containing:
    - `text`: The transcribed speech
    - `prediction`: The classification result
    - `probabilities`: Scores for each possible class

### File Descriptions

- **webapp.html**: Frontend web interface for recording and displaying results
- **api.py**: Flask server handling API requests and audio processing
- **analyze_audio.py**: Core analysis functions for transcription and classification
- **main.py**: Contains utility functions for audio conversion and transcription
- **voice_text_classifier.joblib**: Pre-trained machine learning model

## Troubleshooting

- **Microphone Access**: Ensure your browser has permission to access your microphone
- **API Connection**: If results aren't loading, verify the API server is running at http://localhost:5000
- **Console Errors**: Check your browser's developer console (F12) for error messages
- **File Format**: If uploading files, ensure they are in a supported audio format

## Further Development

Possible enhancements for the project:

- Add user authentication to store and retrieve recordings
- Implement batch processing for multiple audio files
- Expand the classification model with more categories
- Add support for different languages
- Create a progressive web app (PWA) for offline use

## License

[Specify license information here]

## Acknowledgments

- OpenAI for the Whisper speech recognition model
- Flask for the web framework
- scikit-learn for the classification tools