import os
import sys
import joblib
from main import transcribe_with_whisper, convert_to_wav  # Import functions from main.py

def classify_text(text, model_path='voice_text_classifier.joblib'):
    """Classify the transcribed text using the saved model."""
    try:
        # Load the trained model
        classifier = joblib.load(model_path)
        
        # Make prediction
        prediction = classifier.predict([text])[0]
        probabilities = classifier.predict_proba([text])[0]
        
        # Get probability scores
        class_probabilities = {
            label: prob for label, prob in zip(classifier.classes_, probabilities)
        }
        
        return {
            'prediction': prediction,
            'probabilities': class_probabilities,
            'text': text
        }
    except Exception as e:
        print(f"Error in classification: {e}")
        return None

def process_audio_file(audio_path, model_path='voice_text_classifier.joblib', whisper_model_size='tiny'):
    """Process audio file: transcribe and classify."""
    print(f"Processing audio file: {audio_path}")
    
    # Convert to WAV if necessary
    wav_path = convert_to_wav(audio_path)
    if not wav_path:
        return None
    
    print("Transcribing audio...")
    transcription = transcribe_with_whisper(wav_path, model_size=whisper_model_size)
    if not transcription:
        return None
    
    # Classify transcription
    print("Classifying transcription...")
    result = classify_text(transcription, model_path)
    
    return result

def main():
    # Check if an audio file is provided as argument
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        # If the path is relative, make it absolute
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(os.getcwd(), audio_path)
    else:
        # Default audio file if none provided
        audio_path = "D:/Jayanth/AI_Hackathon/audio_sample_2.wav"
        print("No audio file specified. Using default:", audio_path)
    
    # Process the audio file
    result = process_audio_file(audio_path)
    
    if result:
        print("\nResults:")
        print("-" * 50)
        print(f"Transcribed Text: {result['text']}")
        print(f"\nClassification: {result['prediction']}")
        print("\nProbability Scores:")
        for label, prob in result['probabilities'].items():
            print(f"{label}: {prob:.4f}")
    else:
        print("Failed to process audio file")

if __name__ == "__main__":
    main()
