from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from analyze_audio import process_audio_file

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_filepath = temp_file.name
        audio_file.save(temp_filepath)
    
    try:
        # Process the audio file using analyze_audio.py
        result = process_audio_file(temp_filepath)
        
        # Clean up temporary file
        os.unlink(temp_filepath)
        
        if not result:
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        return jsonify(result)
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_filepath):
            os.unlink(temp_filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
