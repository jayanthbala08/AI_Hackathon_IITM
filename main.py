import speech_recognition as sr
import pyaudio
import wave
import os
import argparse
from faster_whisper import WhisperModel
import whisper
from pydub import AudioSegment

def record_audio(filename, duration=5, sample_rate=16000, channels=1):
    chunk = 1024
    format = pyaudio.paInt16
    
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    print(f"Recording for {duration} seconds...")
    
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Recording finished.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    return filename

def convert_to_wav(input_path, output_path="converted_audio.wav"):
    """Convert audio file to WAV format."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error converting file to WAV: {e}")
        return None

def transcribe_with_whisper(audio_path, model_size="tiny"):
    """Transcribe audio using a lightweight Whisper model."""
    try:
        try:
            model = WhisperModel(model_size)
            segments, info = model.transcribe(audio_path, beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
            return transcription
        except ImportError:
            model = whisper.load_model(model_size)
            result = model.transcribe(audio_path)
            return result["text"]
    except Exception as e:
        return f"Error transcribing with Whisper: {e}"

def main():
    parser = argparse.ArgumentParser(description='Speech-to-Text Conversion')
    parser.add_argument('--mode', type=str, choices=['record', 'file'], default='record',
                      help='Mode for audio input: record (record from microphone) or file (use existing file)')
    parser.add_argument('--audio_file', type=str, default=None,
                      help='Path to audio file (required if mode is "file")')
    parser.add_argument('--duration', type=int, default=5,
                      help='Duration of recording in seconds (if recording from microphone)')
    parser.add_argument('--whisper_model', type=str, default='tiny',
                      help='Whisper model size to use (tiny, base, small, medium, large)')
    
    args = parser.parse_args()
    
    # Get audio file
    if args.mode == 'record':
        audio_path = "recorded_audio.wav"
        try:
            record_audio(audio_path, duration=args.duration)
        except Exception as e:
            print(f"Error recording audio: {e}")
            return
    elif args.mode == 'file':
        if not args.audio_file or not os.path.exists(args.audio_file):
            print("Error: Valid audio file path must be provided in 'file' mode.")
            return
        # Convert to WAV if necessary
        if not args.audio_file.lower().endswith(".wav"):
            print("Converting audio file to WAV format...")
            audio_path = convert_to_wav(args.audio_file)
            if not audio_path:
                return
        else:
            audio_path = args.audio_file
    else:
        print("Invalid mode. Please choose 'record' or 'file'.")
        return
    
    # Transcribe
    print(f"Transcribing with Whisper ({args.whisper_model} model)...")
    transcription = transcribe_with_whisper(audio_path, model_size=args.whisper_model)
    
    print("\nTranscription:")
    print(transcription)

if __name__ == "__main__":
    main()
