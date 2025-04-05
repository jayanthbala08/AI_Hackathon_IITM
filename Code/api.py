from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import numpy as np
import json
from analyze_audio import process_audio_file
from generative_response_bot import ResponseGenerator
import gtts
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize the response generator
response_generator = ResponseGenerator(fraud_threshold=0.5)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Configure Flask to use our custom JSON encoder
app.json_encoder = NumpyEncoder

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
        
        # Convert NumPy types to Python native types before generating response
        # This ensures all data is JSON serializable
        sanitized_result = json.loads(json.dumps(result, cls=NumpyEncoder))
        
        # Generate bot response based on analysis
        response_data = response_generator.generate_response(sanitized_result)
        
        # Combine the analysis result with the bot response
        sanitized_result.update({
            'bot_response': response_data['response_text'],
            'is_fraud': response_data['is_fraud'],
            'fraud_probability': response_data.get('fraud_probability', 0),
            'extracted_information': response_data.get('extracted_information', {}),
            'conversation_state': response_data.get('conversation_state', {})
        })
        
        # Generate TTS audio URL for the response
        sanitized_result['tts_url'] = f"/tts?text={response_data['response_text']}"
        
        return jsonify(sanitized_result)
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_filepath):
            os.unlink(temp_filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['GET'])
def text_to_speech():
    """Convert text to speech and return audio file"""
    text = request.args.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Use gTTS to convert text to speech
        tts = gtts.gTTS(text=text, lang='en', slow=False)
        
        # Save to a BytesIO object instead of a file
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Return the audio file
        return send_file(
            fp, 
            mimetype='audio/mp3',
            as_attachment=True,
            download_name='response.mp3'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
