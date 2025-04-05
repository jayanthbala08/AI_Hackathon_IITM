REM filepath: d:\Jayanth\AI_Hackathon\transcribe_directory.bat
@echo off

REM Usage: transcribe_directory.bat <directory_path> <output_csv> <whisper_model>
REM Example: transcribe_directory.bat spam_dataset\audiodataset\fraud fraud_transcriptions.csv tiny

set "DIRECTORY=%1"
set "OUTPUT_CSV=%2"
set "WHISPER_MODEL=%3"

if "%DIRECTORY%"=="" (
    echo Usage: %~nx0 ^<directory_path^> ^<output_csv^> [whisper_model]
    exit /b 1
)

if "%OUTPUT_CSV%"=="" (
    echo Usage: %~nx0 ^<directory_path^> ^<output_csv^> [whisper_model]
    exit /b 1
)

if "%WHISPER_MODEL%"=="" (
    set "WHISPER_MODEL=tiny"
)

if not exist "%DIRECTORY%" (
    echo Error: Directory "%DIRECTORY%" does not exist.
    exit /b 1
)

echo Transcribing audio files in directory: "%DIRECTORY%"
echo Saving transcriptions to: "%OUTPUT_CSV%"
echo Using Whisper model: "%WHISPER_MODEL%"

REM Run the Python script with the directory mode
python main.py --mode directory --directory "%DIRECTORY%" --output_csv "%OUTPUT_CSV%" --whisper_model "%WHISPER_MODEL%"

if %ERRORLEVEL%==0 (
    echo Transcription completed successfully.
) else (
    echo Error occurred during transcription.
)