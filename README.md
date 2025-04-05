# Speech-to-Text Conversion

This program converts speech to text using the Whisper model. It supports two modes:
1. **Record audio**: Record audio from the microphone.
2. **Use an existing audio file**: Provide a pre-recorded audio file for transcription.

## Requirements

Ensure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

## Usage

### Using a Pre-recorded Audio File

To transcribe an existing audio file, use the `--mode file` option and provide the path to the audio file using `--audio_file`. For example:

```bash
python main.py --mode file --audio_file audio_sample_1.wav
```

### Options

- `--mode`: Specify the mode of input. Use `file` for pre-recorded audio files.
- `--audio_file`: Path to the audio file (required if `--mode file` is selected).
- `--whisper_model`: Specify the Whisper model size to use. Options are `tiny`, `base`, `small`, `medium`, and `large`. Default is `tiny`.

### Example

```bash
python main.py --mode file --audio_file path/to/your_audio_file.wav --whisper_model base
```

This will transcribe the audio file using the Whisper model and print the transcription to the console.

## Notes

- Ensure the audio file is in a supported format (e.g., `.wav`).
- For best results, use clear audio with minimal background noise.