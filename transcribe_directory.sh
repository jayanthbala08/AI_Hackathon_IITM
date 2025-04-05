#!/bin/bash

# Usage: ./transcribe_directory.sh <directory_path> <output_csv> <whisper_model>
# Example: ./transcribe_directory.sh spam_dataset/audiodataset/fraud fraud_transcriptions.csv tiny

DIRECTORY=$1
OUTPUT_CSV=$2
WHISPER_MODEL=${3:-tiny}  # Default to "tiny" if not provided

if [ -z "$DIRECTORY" ] || [ -z "$OUTPUT_CSV" ]; then
  echo "Usage: $0 <directory_path> <output_csv> [whisper_model]"
  exit 1
fi

if [ ! -d "$DIRECTORY" ]; then
  echo "Error: Directory '$DIRECTORY' does not exist."
  exit 1
fi

echo "Transcribing audio files in directory: $DIRECTORY"
echo "Saving transcriptions to: $OUTPUT_CSV"
echo "Using Whisper model: $WHISPER_MODEL"

# Run the Python script with the directory mode
python main.py --mode directory --directory "$DIRECTORY" --output_csv "$OUTPUT_CSV" --whisper_model "$WHISPER_MODEL"

if [ $? -eq 0 ]; then
  echo "Transcription completed successfully."
else
  echo "Error occurred during transcription."
fi
