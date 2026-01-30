from flask import Flask, request, jsonify
import os
import google.generativeai as genai
import google.api_core.exceptions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import configparser
import re
import time
from collections import deque
import logging
import sys
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration Loader ---
class ConfigManager:
    """Handles loading configuration from config.ini."""
    def __init__(self, filename="config.ini"):
        self.config = configparser.ConfigParser()
        if not os.path.exists(filename):
            print(f"FATAL ERROR: Configuration file '{filename}' not found. Please create it.")
            sys.exit(f"Error: Configuration file '{filename}' not found.")
        self.config.read(filename)
        self._validate_config()

    def get(self, section, key, fallback=None):
        return self.config.get(section, key, fallback=fallback)

    def getint(self, section, key, fallback=None):
        try:
            return self.config.getint(section, key, fallback=fallback)
        except (ValueError, TypeError):
             print(f"Warning: Invalid integer value for [{section}]{key}. Using fallback: {fallback}")
             return fallback

    def getfloat(self, section, key, fallback=None):
        try:
             return self.config.getfloat(section, key, fallback=fallback)
        except (ValueError, TypeError):
            print(f"Warning: Invalid float value for [{section}]{key}. Using fallback: {fallback}")
            return fallback

    def getboolean(self, section, key, fallback=False):
        try:
            return self.config.getboolean(section, key, fallback=fallback)
        except (ValueError, TypeError):
            print(f"Warning: Invalid boolean value for [{section}]{key}. Using fallback: {fallback}")
            return fallback

    def _validate_config(self):
         """Basic validation for essential config values."""
         api_key = self.get("API", "gemini_api_key")
         if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
             print("FATAL ERROR: Gemini API Key is missing or invalid in config.ini. Please update it.")
             sys.exit("Error: Invalid or missing API Key in config.ini")

# --- Logging Setup ---
log_file = "mindsync_api_log.txt"
log_level = logging.INFO
log_format = '%(asctime)s - %(levelname)s - %(message)s'

logging.basicConfig(level=log_level,
                    format=log_format,
                    handlers=[
                        logging.FileHandler(log_file, encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# --- Global Variables & Initialization ---
try:
    config = ConfigManager()
except SystemExit:
     raise RuntimeError("Configuration error prevented application startup.")

# --- Load Settings from Config ---
API_KEY = config.get("API", "gemini_api_key")
MODEL_NAME = config.get("API", "model_name", fallback="gemini-1.5-flash")
TEMPERATURE = config.getfloat("API", "temperature", fallback=0.7)

MAX_HISTORY_TURNS = config.getint("Settings", "max_history_turns", fallback=6)
MAX_HISTORY_LEN = MAX_HISTORY_TURNS * 2
RATE_LIMIT_MESSAGES = config.getint("Settings", "rate_limit_messages", fallback=5)
RATE_LIMIT_SECONDS = config.getint("Settings", "rate_limit_seconds", fallback=60)
NEGATIVE_SENTIMENT_THRESHOLD = config.getfloat("Settings", "negative_sentiment_threshold", fallback=-0.7)
LOG_FEEDBACK_TO_FILE = config.getboolean("Settings", "log_feedback_to_file", fallback=True)
FEEDBACK_LOG_FILE = config.get("Settings", "feedback_log_file", fallback="feedback_log.txt")

# --- AI Model Configuration ---
try:
    genai.configure(api_key=API_KEY)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    generation_config = genai.types.GenerationConfig(temperature=TEMPERATURE)
    model = genai.GenerativeModel(
        MODEL_NAME,
        safety_settings=safety_settings,
        generation_config=generation_config
    )
    logging.info(f"Gemini model '{MODEL_NAME}' configured successfully.")
except Exception as e:
    logging.exception("FATAL ERROR: Failed to configure AI model.")
    raise RuntimeError(f"AI Model Configuration Error: {e}")

# --- Other Initializations ---
analyzer = SentimentIntensityAnalyzer()
# Store chat histories by session ID
chat_histories = {}
message_timestamps = {}
last_bot_messages = {}

# India-Specific Emergency Resources
crisis_message = (
    "I hear your pain, like a storm that seems unending. In moments like these, a compassionate human voice can be the lighthouse we need.\n\n"
    "**Please reach out to these 24/7 helplines in India:**\n\n"
    "* **Vandrevala Foundation:** +91 9999 666 555\n"
    "* **KIRAN (Govt. of India):** 1800-599-0019\n"
    "* **AASRA:** +91-9820466726\n"
    "* **iCALL (TISS):** 022-25521111 (Mon-Sat, 10 AM - 8 PM)\n\n"
    "Remember, even in the darkest night, you are not alone. There are hands waiting to help you through this moment."
)

initial_greeting_text = "Namaste! I'm MindSync, your AI companion for mental wellness. In our conversation, I'll share thoughtful suggestions for breathing exercises, music, movies, and activities - one at a time. How are you feeling today?"

# --- Helper Functions ---
def _is_rate_limited(session_id):
    """Checks if the user is sending messages too quickly."""
    if session_id not in message_timestamps:
        message_timestamps[session_id] = deque()
    
    now = time.time()
    while message_timestamps[session_id] and message_timestamps[session_id][0] < now - RATE_LIMIT_SECONDS:
        message_timestamps[session_id].popleft()
    
    if len(message_timestamps[session_id]) >= RATE_LIMIT_MESSAGES:
        return True
    
    message_timestamps[session_id].append(now)
    return False

def _check_for_crisis(user_input, sentiment_score):
    """Checks for crisis keywords OR very negative sentiment."""
    text_lower = user_input.lower()
    crisis_keywords_patterns = [
        r'\bsuicide\b', r'\bkill myself\b', r'\bwant to die\b', r'\bend my life\b', r'\bending it all\b',
        r'\bhopeless\b', r'\bcan\'t go on\b', r'\bno reason to live\b', r'\bgive up\b', r'\bno point\b',
        r'\bself harm\b', r'\bself-harm\b', r'\bhurting myself\b', r'\bcut myself\b', r'\boverdose\b',
        r'\bcan\'t cope\b', r'\bextreme despair\b', r'\bseverely depressed\b', r'\bunbearable pain\b',
        r'\bmake it stop\b'
    ]
    for pattern in crisis_keywords_patterns:
        if re.search(pattern, text_lower):
            logging.warning(f"Crisis keyword match detected: {pattern}")
            return True
    if sentiment_score <= NEGATIVE_SENTIMENT_THRESHOLD:
        logging.warning(f"Very negative sentiment detected: {sentiment_score:.2f}")
        return True
    return False

def _build_prompt(user_input, pulse_state, sentiment_score, history_list):
    """Constructs the detailed prompt for the Gemini AI. Takes history list."""
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['parts'][0]}" for msg in history_list])
    
    # Determine which suggestion type to provide based on conversation history
    suggestion_types = ["breathing", "music", "movie", "activity"]
    
    # Count how many messages in history to determine which suggestion to give
    message_count = len([msg for msg in history_list if msg['role'] == 'model'])
    suggestion_index = message_count % 4
    current_suggestion = suggestion_types[suggestion_index]
    
    prompt = f"""
**Persona:** You are 'MindSync', an AI-powered Mental Wellness Companion designed as a warm guide and active listener. Your primary role is to address the lack of accessible mental health support in India by providing empathetic conversation, helpful resources, and coping strategies. Be exceptionally patient, understanding, non-judgmental, and culturally aware (within AI limits). Your tone should be consistently supportive and encouraging.

**Mission Context:** Mental health faces stigma and accessibility challenges in India. Be a safe, confidential (within AI limits) first point of contact. Encourage well-being and gently guide towards professional help when appropriate, normalizing seeking help.

**Core Instructions:**

1.  **KEEP RESPONSES CONCISE AND POETIC:** Your responses must be brief, focused, and beautifully worded. Use metaphors, imagery and thoughtful language. Maximum 3-4 short paragraphs total.

2.  **Beautiful Empathy & Validation:** Start with a thoughtful, poetic acknowledgment of the user's feelings in 2-3 elegant sentences. Use imagery and metaphors when appropriate.

3.  **Gentle Severity Assessment (Internal Use Only):** Based on language, sentiment ({sentiment_score=:.2f}), and conversation history, internally gauge distress level. **DO NOT explicitly state your assessment.** Use it *only* to tailor response tone.

4.  **PROVIDE ONLY ONE SPECIFIC SUGGESTION PER RESPONSE:** In this response, you should ONLY provide a suggestion for: {current_suggestion}

    * If {current_suggestion} == "breathing": Provide ONE specific, named breathing technique with brief steps (e.g., "4-7-8 Breathing: Inhale for 4 counts, hold for 7, exhale for 8")
    * If {current_suggestion} == "music": Recommend ONE specific song or artist with a beautiful description of why it might help (e.g., "The gentle melodies of 'Raag Bhairavi' by Pandit Ravi Shankar can wash over you like a soothing stream...")
    * If {current_suggestion} == "movie": Recommend ONE specific movie title with a poetic description of its mood or message (e.g., "'Zindagi Na Milegi Dobara' captures the essence of freedom and friendship like golden sunlight on an open road...")
    * If {current_suggestion} == "activity": ONE simple, immediate activity described in an inspiring way (e.g., "Consider a mindful walk where each step connects you to the earth beneath you, grounding your thoughts like roots of a mighty tree...")

5.  **Format Your Response in Clear Sections:**
    * Beautiful, poetic acknowledgment of user's feelings (2-3 sentences)
    * ONE specific suggestion presented elegantly
    * Brief closing encouragement with imagery (1-2 sentences)

6.  **Integrate Pulse Info (Elegantly):** If provided (e.g., '{pulse_state}'), acknowledge with a poetic touch in ONE sentence maximum.

7.  **--- CRITICAL BOUNDARIES ---**
    * **NO DIAGNOSIS - EVER:** **Under NO circumstances** suggest or confirm *any* mental health diagnosis.
    * **NOT A THERAPIST:** Explicitly state you are an AI if the user seems confused. Do not replace professional help.
    * **GENTLE PROFESSIONAL REDIRECTION:** For severe cases, briefly suggest professional help.

8.  **Response Style:** Warm, hopeful, poetic language. NO LONG PARAGRAPHS. Use beautiful imagery and metaphors.

**Conversation History (for context):**
{formatted_history}

**Current User Input:** "{user_input}"
"""
    # Add pulse info only if present
    if pulse_state and pulse_state.strip():
        prompt += f"\n**User's Physical State/Pulse Description:** \"{pulse_state}\""

    # Final instruction for the AI's output format
    prompt += f"\n\n**MindSync's Supportive Response (KEEP IT BRIEF, BEAUTIFUL, AND FOCUSED ON {current_suggestion.upper()} SUGGESTION ONLY):**"
    return prompt

def _update_internal_history(session_id, role, text):
    """Appends message to internal history and prunes if needed."""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    chat_histories[session_id].append({'role': role, 'parts': [text]})
    
    # Keep history trimmed
    while len(chat_histories[session_id]) > MAX_HISTORY_LEN:
        chat_histories[session_id].pop(0)  # Remove oldest message

def _get_ai_response(session_id, user_input, pulse_state, sentiment_score):
    """Calls the Gemini API and handles responses/errors."""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
        # Add initial greeting
        _update_internal_history(session_id, 'model', initial_greeting_text)
    
    prompt = _build_prompt(user_input, pulse_state, sentiment_score, chat_histories[session_id])
    ai_response_text = "Sorry, I wasn't able to generate a response. Please try again."  # Default error

    try:
        # Call Gemini API
        convo = model.start_chat(history=chat_histories[session_id])
        response = convo.send_message(prompt)

        # Process Response
        if response.parts:
            ai_response_text = response.text
            if "cannot fulfill this request" in ai_response_text.lower():
                logging.warning("AI indicated inability to fulfill request.")
                ai_response_text = "I apologize, but I cannot fulfill that specific request. Could we talk about something else?"
        else:
            # Handle cases where the response might be blocked by safety settings
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'N/A'
            logging.warning(f"AI response blocked or empty. Finish Reason: {block_reason}")
            ai_response_text = "I'm sorry, I encountered an issue generating a response for that topic due to safety guidelines. Perhaps we could try a different subject?"

        # Store for feedback mechanism
        last_bot_messages[session_id] = ai_response_text
        _update_internal_history(session_id, 'model', ai_response_text)
        return ai_response_text, "Status: Ready"

    except google.api_core.exceptions.ResourceExhausted as e:
        error_message = "API Quota Exceeded. Please check your Google Cloud quota or try again later."
        logging.error(f"API Error: {e}")
        return error_message, f"Status: Error - {error_message}"
    except google.api_core.exceptions.GoogleAPIError as e:
        error_message = f"An API error occurred: {e}. Please try again."
        logging.error(f"API Error: {e}")
        return error_message, f"Status: Error - {error_message}"
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}. Please check logs or try again."
        logging.exception("Unexpected error during AI response generation.")
        return error_message, f"Status: Error - Unexpected error"

# --- API Routes ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id', 'default')
    pulse_state = data.get('pulse_state', '')
    
    if not user_input.strip():
        return jsonify({'status': 'error', 'message': 'Empty message'})
    
    # Rate Limiting
    if _is_rate_limited(session_id):
        return jsonify({
            'status': 'error',
            'message': 'Too many messages. Please wait a moment before sending more.'
        })
    
    # Sentiment & Crisis Check
    sentiment_score = analyzer.polarity_scores(user_input)['compound']
    is_potentially_crisis = _check_for_crisis(user_input, sentiment_score)
    
    # Update internal history
    _update_internal_history(session_id, 'user', user_input)
    
    if is_potentially_crisis:
        logging.warning("Crisis message triggered.")
        last_bot_messages[session_id] = crisis_message
        _update_internal_history(session_id, 'model', crisis_message)
        return jsonify({
            'status': 'success',
            'message': crisis_message,
            'is_crisis': True
        })
    
    # Get AI Response
    ai_response, status = _get_ai_response(session_id, user_input, pulse_state, sentiment_score)
    
    return jsonify({
        'status': 'success',
        'message': ai_response,
        'is_crisis': False
    })

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in chat_histories:
        chat_histories[session_id] = []
    
    if session_id in message_timestamps:
        message_timestamps[session_id] = deque()
    
    if session_id in last_bot_messages:
        last_bot_messages[session_id] = ""
    
    # Add initial greeting back
    _update_internal_history(session_id, 'model', initial_greeting_text)
    
    return jsonify({
        'status': 'success',
        'message': 'Chat cleared',
        'initial_message': initial_greeting_text
    })

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    session_id = data.get('session_id', 'default')
    positive_feedback = data.get('positive', True)
    
    rating = "👍 Positive" if positive_feedback else "👎 Negative"
    
    if session_id not in last_bot_messages or not last_bot_messages[session_id]:
        return jsonify({
            'status': 'error',
            'message': 'No previous message to provide feedback on'
        })
    
    feedback_log_entry = f"Feedback Received: {rating} for message:\n---\n{last_bot_messages[session_id]}\n---\n"
    
    # Log to console via logging module
    logging.info(f"Feedback ({rating}) recorded.")
    logging.debug(feedback_log_entry)
    
    # Optional: Log to a separate feedback file
    if LOG_FEEDBACK_TO_FILE:
        try:
            with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {feedback_log_entry}\n")
        except Exception as e:
            logging.error(f"Error writing feedback log to {FEEDBACK_LOG_FILE}: {e}")
    
    return jsonify({
        'status': 'success',
        'message': f"Feedback ({rating}) recorded. Thank you!"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)