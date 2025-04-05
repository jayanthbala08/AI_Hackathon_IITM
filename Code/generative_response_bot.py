import os
import sys
import time
import json
import random
import argparse
import numpy as np
from datetime import datetime
from main import record_audio, transcribe_with_whisper, convert_to_wav
from analyze_audio import process_audio_file, classify_text

class ResponseGenerator:
    """Generate responses based on audio classification results."""
    
    def __init__(self, fraud_threshold=0.5, templates_file=None):
        """
        Initialize the response generator.
        
        Args:
            fraud_threshold: Probability threshold to classify as fraud (default: 0.5)
            templates_file: Path to JSON file with response templates
        """
        self.fraud_threshold = fraud_threshold
        self.conversation_state = {
            "greeted": False,
            "asked_for_info": False,
            "collected_name": False,
            "collected_contact": False,
            "collected_purpose": False,
            "farewell_given": False
        }
        
        # Load templates from file if provided, otherwise use defaults
        if templates_file and os.path.exists(templates_file):
            with open(templates_file, 'r') as f:
                self.templates = json.load(f)
        else:
            self.templates = self._get_default_templates()
    
    def _get_default_templates(self):
        """Get default response templates."""
        return {
            "fraud_responses": [
                "This appears to be a fraudulent call. I cannot proceed with this conversation.",
                "Our systems have flagged this conversation as potentially fraudulent. This call will be reported.",
                "Warning: This conversation has been identified as a potential scam. Please be aware that we monitor all communications for fraud."
            ],
            "greetings": [
                "Hello! How can I assist you today?",
                "Good day! Thank you for contacting us. How may I help you?",
                "Welcome! I'm here to help with your inquiry."
            ],
            "info_requests": [
                "Could you please tell me your name and how I can help you today?",
                "To better assist you, may I have your name and a brief description of your query?",
                "I'd be happy to help. Could you share your name and what you're looking for assistance with?"
            ],
            "contact_requests": [
                "Thank you. Could you also provide a contact number or email where we can reach you?",
                "Great. To proceed, I'll need a phone number or email address for our records.",
                "To continue assisting you, could you share a phone number or email address?"
            ],
            "clarification_requests": [
                "Could you please provide more details about your request?",
                "I'd like to understand your needs better. Can you elaborate on that?",
                "To ensure I assist you correctly, could you explain more about what you need?"
            ],
            "acknowledgments": [
                "I understand. Let me help you with that.",
                "Thank you for sharing that information.",
                "I appreciate your patience. I'm here to assist you."
            ],
            "farewells": [
                "Thank you for contacting us. Is there anything else I can help you with today?",
                "I appreciate your time today. Please don't hesitate to reach out if you need further assistance.",
                "Thank you for your inquiry. Have a wonderful day!"
            ]
        }
    
    def _is_fraud(self, classification_result):
        """Determine if the message is fraudulent based on classification probability."""
        fraud_prob = 0
        
        # Check for fraud-related labels in probabilities
        for label, prob in classification_result['probabilities'].items():
            if 'fraud' in label.lower() or 'scam' in label.lower():
                fraud_prob = max(fraud_prob, prob)
        
        return fraud_prob >= self.fraud_threshold, fraud_prob
    
    def _extract_information(self, text):
        """Extract potentially useful information from text."""
        extracted_info = {}
        
        # Simple name extraction
        name_patterns = ["my name is", "i am", "this is", "call me", "speaking"]
        for pattern in name_patterns:
            if pattern in text.lower():
                idx = text.lower().find(pattern) + len(pattern)
                name_candidate = text[idx:idx+30].strip().split('.')[0].split(',')[0].split('and')[0]
                if name_candidate and len(name_candidate) > 2 and len(name_candidate) < 30:
                    extracted_info['name'] = name_candidate.strip()
                    self.conversation_state["collected_name"] = True
                    break
        
        # Contact info extraction
        # Phone numbers
        import re
        phone_pattern = r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b'
        phones = re.findall(phone_pattern, text)
        if phones:
            # Format the first match as a phone number
            phone_parts = [part for part in phones[0] if part]
            if phone_parts:
                if len(phone_parts) >= 3:
                    extracted_info['phone'] = '-'.join(phone_parts[-3:])
                    self.conversation_state["collected_contact"] = True
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            extracted_info['email'] = emails[0]
            self.conversation_state["collected_contact"] = True
        
        # Purpose/query extraction (simple approach)
        purpose_indicators = ["need", "want", "looking for", "help with", "question about", "interested in"]
        for indicator in purpose_indicators:
            if indicator in text.lower():
                idx = text.lower().find(indicator)
                end_idx = text.find(".", idx)
                if end_idx == -1:
                    end_idx = len(text)
                purpose = text[idx:end_idx].strip()
                if purpose and len(purpose) > 5:
                    extracted_info['purpose'] = purpose
                    self.conversation_state["collected_purpose"] = True
                    break
        
        return extracted_info
    
    def _select_random_response(self, category):
        """Select a random response from the specified template category."""
        if category in self.templates and self.templates[category]:
            return random.choice(self.templates[category])
        return "I'm processing your request."
    
    def generate_response(self, classification_result):
        """
        Generate appropriate response based on classification results.
        
        Args:
            classification_result: Dictionary with classification results
            
        Returns:
            Dictionary with response text and metadata
        """
        text = classification_result['text']
        prediction = classification_result['prediction']
        
        # Check for fraud
        is_fraud, fraud_probability = self._is_fraud(classification_result)
        
        # Extract information from text
        extracted_info = self._extract_information(text)
        
        # Prepare response data
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "input_text": text,
            "classification": prediction,
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "extracted_information": extracted_info,
        }
        
        # Generate response based on fraud status and conversation state
        if is_fraud:
            response_data["response_text"] = self._select_random_response("fraud_responses")
            response_data["action"] = "flag_as_fraud"
        else:
            # Normal conversation flow
            if not self.conversation_state["greeted"]:
                # Initial greeting
                response_data["response_text"] = self._select_random_response("greetings")
                self.conversation_state["greeted"] = True
            
            elif not self.conversation_state["asked_for_info"]:
                # Ask for information
                response_data["response_text"] = self._select_random_response("info_requests")
                self.conversation_state["asked_for_info"] = True
            
            elif not self.conversation_state["collected_name"] and not self.conversation_state["collected_contact"]:
                # If we still need basic information
                if "name" in extracted_info:
                    # If we just got their name, ask for contact info
                    response_data["response_text"] = f"Thank you, {extracted_info['name']}. " + self._select_random_response("contact_requests")
                else:
                    # Still need name
                    response_data["response_text"] = self._select_random_response("info_requests")
            
            elif not self.conversation_state["collected_contact"]:
                # Need contact information
                response_data["response_text"] = self._select_random_response("contact_requests")
            
            elif not self.conversation_state["collected_purpose"]:
                # Need to understand their purpose
                response_data["response_text"] = self._select_random_response("clarification_requests")
            
            else:
                # We have all the information we need
                if not self.conversation_state["farewell_given"]:
                    # Acknowledge and end conversation
                    response_data["response_text"] = self._select_random_response("acknowledgments") + " " + self._select_random_response("farewells")
                    self.conversation_state["farewell_given"] = True
                else:
                    # If they continue after farewell
                    response_data["response_text"] = self._select_random_response("farewells")
        
        # Add conversation state
        response_data["conversation_state"] = self.conversation_state.copy()
        
        return response_data

class ConversationBot:
    """A bot that converses with users through audio messages."""
    
    def __init__(self, model_path='../Models/voice_text_classifier.joblib', 
                 whisper_model_size='tiny', 
                 fraud_threshold=0.5,
                 log_directory='../logs'):
        """
        Initialize the conversation bot.
        
        Args:
            model_path: Path to the classification model
            whisper_model_size: Size of the Whisper model for transcription
            fraud_threshold: Threshold for fraud detection
            log_directory: Directory to save conversation logs
        """
        self.model_path = model_path
        self.whisper_model_size = whisper_model_size
        self.response_generator = ResponseGenerator(fraud_threshold=fraud_threshold)
        
        # Create log directory if it doesn't exist
        self.log_directory = log_directory
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Initialize conversation log
        self.conversation_log = []
        self.session_id = f"session_{int(time.time())}"
    
    def record_and_process(self, duration=5):
        """
        Record audio from microphone and process it.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Dictionary with processing results and response
        """
        print("\nListening for your message...")
        audio_path = "temp_recording.wav"
        
        try:
            # Record audio
            record_audio(audio_path, duration=duration)
            
            # Process the audio
            result = process_audio_file(
                audio_path, 
                model_path=self.model_path,
                whisper_model_size=self.whisper_model_size
            )
            
            if not result:
                return {
                    "status": "error",
                    "message": "Failed to process audio",
                    "response_text": "I couldn't process your audio. Could you please repeat that?"
                }
            
            # Generate response
            response_data = self.response_generator.generate_response(result)
            
            # Log the interaction
            self.log_interaction(result, response_data)
            
            return response_data
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "response_text": "Sorry, I encountered an error. Please try again."
            }
    
    def process_existing_audio(self, audio_path):
        """
        Process an existing audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with processing results and response
        """
        try:
            # Process the audio
            result = process_audio_file(
                audio_path, 
                model_path=self.model_path,
                whisper_model_size=self.whisper_model_size
            )
            
            if not result:
                return {
                    "status": "error",
                    "message": f"Failed to process audio file: {audio_path}",
                    "response_text": "I couldn't process this audio file."
                }
            
            # Generate response
            response_data = self.response_generator.generate_response(result)
            
            # Log the interaction
            self.log_interaction(result, response_data)
            
            return response_data
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "response_text": "Sorry, I encountered an error processing this audio file."
            }
    
    def log_interaction(self, result, response_data):
        """Log the interaction for analysis."""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transcription": result["text"],
            "classification": result["prediction"],
            "probabilities": {str(k): float(v) for k, v in result["probabilities"].items()},
            "response": response_data["response_text"],
            "is_fraud": response_data["is_fraud"],
            "extracted_information": response_data.get("extracted_information", {})
        }
        
        self.conversation_log.append(log_entry)
        
        # Save the log entry
        log_file = os.path.join(self.log_directory, f"{self.session_id}.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def save_conversation_summary(self):
        """Save a summary of the conversation."""
        if not self.conversation_log:
            return
            
        summary_file = os.path.join(self.log_directory, f"{self.session_id}_summary.json")
        
        summary = {
            "session_id": self.session_id,
            "start_time": self.conversation_log[0]["timestamp"],
            "end_time": self.conversation_log[-1]["timestamp"],
            "messages_count": len(self.conversation_log),
            "fraud_messages_count": sum(1 for entry in self.conversation_log if entry["is_fraud"]),
            "extracted_information": {},
        }
        
        # Combine all extracted information
        for entry in self.conversation_log:
            for key, value in entry.get("extracted_information", {}).items():
                if key not in summary["extracted_information"]:
                    summary["extracted_information"][key] = value
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"Conversation summary saved to {summary_file}")

def main():
    """Main function to run the conversation bot."""
    parser = argparse.ArgumentParser(description='Audio Conversation Bot')
    parser.add_argument('--mode', type=str, choices=['interactive', 'file'], default='interactive',
                      help='Bot mode: interactive (microphone) or file (process existing audio)')
    parser.add_argument('--audio_file', type=str, default=None,
                      help='Path to audio file (required if mode is "file")')
    parser.add_argument('--duration', type=int, default=5,
                      help='Duration of recording in seconds (if interactive mode)')
    parser.add_argument('--whisper_model', type=str, default='tiny',
                      help='Whisper model size to use (tiny, base, small, medium, large)')
    parser.add_argument('--model_path', type=str, default='../Models/voice_text_classifier.joblib',
                      help='Path to the classification model')
    parser.add_argument('--fraud_threshold', type=float, default=0.5,
                      help='Threshold probability to classify as fraud')
    
    args = parser.parse_args()
    
    # Initialize the bot
    bot = ConversationBot(
        model_path=args.model_path,
        whisper_model_size=args.whisper_model,
        fraud_threshold=args.fraud_threshold
    )
    
    try:
        if args.mode == 'interactive':
            print("Starting interactive conversation bot. Press Ctrl+C to exit.")
            print("=" * 50)
            
            while True:
                # Record and process
                result = bot.record_and_process(duration=args.duration)
                
                # Print the response
                print("\nBot:", result["response_text"])
                
                # If it's fraud, we might want to end the conversation
                if result.get("is_fraud", False):
                    print("\nFraud detected. Conversation flagged.")
                    user_continue = input("\nDo you want to continue the conversation? (y/n): ")
                    if user_continue.lower() != 'y':
                        break
                
                # Option to end conversation
                user_continue = input("\nContinue conversation? (y/n): ")
                if user_continue.lower() != 'y':
                    break
                
        elif args.mode == 'file':
            if not args.audio_file:
                print("Error: Audio file path required in file mode.")
                return
                
            # Process the file
            result = bot.process_existing_audio(args.audio_file)
            
            # Print the results
            print("\nResults:")
            print("-" * 50)
            if "input_text" in result:
                print(f"Transcribed Text: {result['input_text']}")
            
            print(f"\nBot Response: {result['response_text']}")
            
            if result.get("is_fraud", False):
                print("\nWarning: This message was classified as potentially fraudulent!")
                print(f"Fraud probability: {result.get('fraud_probability', 0):.4f}")
            
            if "extracted_information" in result and result["extracted_information"]:
                print("\nExtracted Information:")
                for key, value in result["extracted_information"].items():
                    print(f"- {key}: {value}")
    
    except KeyboardInterrupt:
        print("\nConversation ended by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Save conversation summary
        bot.save_conversation_summary()
        print("\nConversation ended. Thank you!")

if __name__ == "__main__":
    main()