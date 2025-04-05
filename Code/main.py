import speech_recognition as sr
import pyaudio
import wave
import os
import argparse
from faster_whisper import WhisperModel
import whisper
from pydub import AudioSegment
import csv
from shutil import which  

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

def is_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible."""
    return which("ffmpeg") is not None

def convert_to_wav(input_path, output_path="converted_audio.wav"):
    """Convert audio file to WAV format."""
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        return None
    if not is_ffmpeg_installed():
        print("Error: ffmpeg is not installed or not in PATH. Please install ffmpeg to proceed.")
        return None
    try:
        print(f"Converting '{input_path}' to WAV format...")
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        print(f"Conversion successful: '{output_path}'")
        return output_path
    except Exception as e:
        print(f"Error converting file '{input_path}' to WAV: {e}")
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

def transcribe_directory_to_csv(directory, output_csv, model_size="tiny"):
    """Transcribe all audio files in a directory and save results to a CSV file."""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    audio_files = [f for f in os.listdir(directory) if f.lower().endswith(('.wav', '.mp3', '.mp4'))]
    if not audio_files:
        print(f"No audio files found in directory '{directory}'.")
        return

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Transcription"])  # Write header

        for audio_file in audio_files:
            audio_path = os.path.join(directory, audio_file)
            print(f"Processing file: {audio_file}")
            print(f"Full path: {audio_path}")

            if not os.path.exists(audio_path):
                print(f"Error: File '{audio_path}' does not exist. Skipping.")
                continue

            # Convert to WAV if necessary
            if not audio_file.lower().endswith(".wav"):
                print("Converting audio file to WAV format...")
                audio_path = convert_to_wav(audio_path)
                if not audio_path:
                    print(f"Skipping file '{audio_file}' due to conversion error.")
                    continue

            # Transcribe
            transcription = transcribe_with_whisper(audio_path, model_size=model_size)
            writer.writerow([audio_file, transcription])  # Write to CSV

    print(f"Transcriptions saved to '{output_csv}'.")

def main():
    parser = argparse.ArgumentParser(description='Speech-to-Text Conversion')
    parser.add_argument('--mode', type=str, choices=['record', 'file', 'directory'], default='record',
                      help='Mode for audio input: record (record from microphone), file (use existing file), or directory (transcribe all files in a directory)')
    parser.add_argument('--audio_file', type=str, default=None,
                      help='Path to audio file (required if mode is "file")')
    parser.add_argument('--directory', type=str, default=None,
                      help='Path to directory containing audio files (required if mode is "directory")')
    parser.add_argument('--output_csv', type=str, default='transcriptions.csv',
                      help='Path to output CSV file (used in "directory" mode)')
    parser.add_argument('--duration', type=int, default=5,
                      help='Duration of recording in seconds (if recording from microphone)')
    parser.add_argument('--whisper_model', type=str, default='tiny',
                      help='Whisper model size to use (tiny, base, small, medium, large)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'record':
            audio_path = "recorded_audio.wav"
            try:
                record_audio(audio_path, duration=args.duration)
            except Exception as e:
                print(f"Error recording audio: {e}")
                return
            print(f"Transcribing with Whisper ({args.whisper_model} model)...")
            transcription = transcribe_with_whisper(audio_path, model_size=args.whisper_model)
            print("\nTranscription:")
            print(transcription)
        elif args.mode == 'file':
            if not args.audio_file or not os.path.exists(args.audio_file):
                print("Error: Valid audio file path must be provided in 'file' mode.")
                return
            if not args.audio_file.lower().endswith(".wav"):
                print("Converting audio file to WAV format...")
                audio_path = convert_to_wav(args.audio_file)
                if not audio_path:
                    return
            else:
                audio_path = args.audio_file
            print(f"Transcribing with Whisper ({args.whisper_model} model)...")
            transcription = transcribe_with_whisper(audio_path, model_size=args.whisper_model)
            print("\nTranscription:")
            print(transcription)
        elif args.mode == 'directory':
            if not args.directory:
                print("Error: Directory path must be provided in 'directory' mode.")
                return
            transcribe_directory_to_csv(args.directory, args.output_csv, model_size=args.whisper_model)
        else:
            print("Invalid mode. Please choose 'record', 'file', or 'directory'.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
