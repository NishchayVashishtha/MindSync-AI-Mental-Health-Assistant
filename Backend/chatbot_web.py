import os
import gradio as gr
import google.generativeai as genai
import google.api_core.exceptions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import configparser # Use configparser instead of dotenv
import re
import time
from collections import deque
import threading # Still needed for feedback logging potentially
import sys
import logging # Use logging module for better feedback/error logging

# --- Configuration Loader ---
class ConfigManager:
    """Handles loading configuration from config.ini."""
    def __init__(self, filename="config.ini"):
        self.config = configparser.ConfigParser()
        if not os.path.exists(filename):
            # Critical error if config is missing
            print(f"FATAL ERROR: Configuration file '{filename}' not found. Please create it.")
            # Attempt to show error in Gradio if possible, otherwise exit
            try:
                gr.Error(f"Configuration file '{filename}' not found. Application cannot start.")
            except Exception:
                pass # Gradio might not be fully initialized yet
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
             try:
                 gr.Error("Gemini API Key is missing or invalid in config.ini. Application cannot start.")
             except Exception:
                 pass
             sys.exit("Error: Invalid or missing API Key in config.ini")

# --- Logging Setup ---
log_file = "mindsync_web_log.txt"
log_level = logging.INFO # Or DEBUG for more details
log_format = '%(asctime)s - %(levelname)s - %(message)s'

logging.basicConfig(level=log_level,
                    format=log_format,
                    handlers=[
                        logging.FileHandler(log_file, encoding='utf-8'),
                        logging.StreamHandler() # Also print to console
                    ])

# --- Global Variables & Initialization ---
try:
    config = ConfigManager()
except SystemExit:
     # ConfigManager handles exit if config is fatally flawed
     # Need to prevent Gradio from fully launching if this happens
     # One way is to raise an exception that stops the script *before* demo.launch()
     raise RuntimeError("Configuration error prevented application startup.")


# --- Load Settings from Config ---
API_KEY = config.get("API", "gemini_api_key")
MODEL_NAME = config.get("API", "model_name", fallback="gemini-1.5-flash") # Or 1.5-pro, etc.
TEMPERATURE = config.getfloat("API", "temperature", fallback=0.7)

MAX_HISTORY_TURNS = config.getint("Settings", "max_history_turns", fallback=6)
MAX_HISTORY_LEN = MAX_HISTORY_TURNS * 2 # Store pairs (user + model)
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
    try:
       gr.Error(f"Failed to configure AI model. Check API key and model name.\nError: {e}")
    except Exception:
        pass
    # Allow Gradio to potentially show the error, but log it critically
    raise RuntimeError(f"AI Model Configuration Error: {e}")


# --- Other Initializations ---
analyzer = SentimentIntensityAnalyzer()
# Stores history in the format Gemini expects: [{'role': 'user'/'model', 'parts': [text]}, ...]
internal_chat_history = []
message_timestamps = deque()
last_bot_message_content_for_feedback = "" # Store only the text for feedback

# India-Specific Emergency Resources (Consider making this configurable)
crisis_message = (
    "I understand you're in immense pain and distress right now. It sounds like a serious crisis situation. "
    "Your safety is the absolute top priority, and as an AI, I am *not* equipped to provide the immediate, professional help you need.\n\n"
    "**Please reach out *immediately* to one of these 24/7 helplines in India:**\n\n"
    "* **Vandrevala Foundation:** +91 9999 666 555\n"
    "* **KIRAN (Govt. of India):** 1800-599-0019\n"
    "* **AASRA:** +91-9820466726\n"
    "* **iCALL (TISS):** 022-25521111 (Mon-Sat, 10 AM - 8 PM)\n\n"
    "**Search online for 'crisis support India' for more local options.**\n\n"
    "Please connect with someone who can support you. You are not alone."
)

initial_greeting_text = "Namaste! I'm MindSync, your AI companion for mental wellness support, focusing on accessible help in India. How are you feeling today? Please remember, I'm an AI and cannot provide medical advice or diagnosis."

# --- Helper Functions ---

def _is_rate_limited():
    """Checks if the user is sending messages too quickly."""
    now = time.time()
    while message_timestamps and message_timestamps[0] < now - RATE_LIMIT_SECONDS:
        message_timestamps.popleft()
    if len(message_timestamps) >= RATE_LIMIT_MESSAGES:
        return True
    message_timestamps.append(now)
    return False

def _check_for_crisis(user_input, sentiment_score):
    """Checks for crisis keywords OR very negative sentiment."""
    text_lower = user_input.lower()
    # Using word boundaries (\b) for more specific matching
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

    prompt = f"""
**Persona:** You are 'MindSync', an AI-powered Mental Wellness Companion designed as a warm guide and active listener. Your primary role is to address the lack of accessible mental health support in India by providing empathetic conversation, helpful resources, and coping strategies. Be exceptionally patient, understanding, non-judgmental, and culturally aware (within AI limits). Your tone should be consistently supportive and encouraging.

**Mission Context:** Mental health faces stigma and accessibility challenges in India. Be a safe, confidential (within AI limits) first point of contact. Encourage well-being and gently guide towards professional help when appropriate, normalizing seeking help.

**Core Instructions:**

1.  **Empathy & Validation (Paramount):** Start by deeply acknowledging the user's feelings. Use phrases reflecting active listening: "I hear that you're feeling [emotion]...", "It sounds like you're going through a really challenging time with [situation]...", "Thank you for trusting me with this, it takes courage to share...". Validate their experience *before* anything else: "It's completely understandable to feel [emotion] in that situation."
2.  **Gentle Severity Assessment (Internal Use Only):** Based on language, sentiment ({sentiment_score=:.2f}), and conversation history, internally gauge distress level. **DO NOT explicitly state your assessment.** Use it *only* to tailor response tone (more soothing for high distress) and suggestion complexity (simpler for high distress).
3.  **Offer Comfort & Hope:** Provide genuine-sounding reassurance ("You're not alone in this," "Things can get better, even if it doesn't feel like it right now," "Be kind to yourself during this difficult time").
4.  **Suggest Contextual Coping Mechanisms (Simple & Actionable):** Offer 1-2 relevant techniques:
    * *Anxiety/Stress:* Box breathing (describe steps), 5-4-3-2-1 grounding (describe), brief mindful body scan, simple affirmations.
    * *Low Mood:* Behavioral activation (one small, achievable pleasant/necessary activity), gratitude reflection (one specific thing), gentle stretching, listening to a specific type of uplifting music.
    * *Overwhelm:* Brain dump (writing thoughts), breaking a task into tiny steps, setting a 5-minute timer for one focus item.
5.  **Provide Relevant Recommendations (Tailored & Diverse - Suggest Types/Examples):**
    * *Activities:* Mindfulness practices, journaling prompts (e.g., "write about one small success today"), creative expression (drawing, music), light physical activity, spending time in nature (even briefly).
    * *Media (Suggest Genres/Themes):* Calming music (Indian classical, nature sounds, ambient), uplifting playlists, light-hearted or inspiring books/movies/shows (mention genres), engaging podcasts on well-being.
    * *Resources (Mention Types & Reputable Indian Examples):*
        * Info: NIMHANS website, The Live Love Laugh Foundation, other credible Indian NGOs.
        * Apps: Mention features (guided meditation, mood tracking) and *types* of apps.
        * Support: Concept of online moderated communities (use caution), Peer support groups (concept).
        * Professional Help: **Concept of Tele-Counseling** platforms in India as an accessible option.
6.  **Integrate Pulse Info (Subtly):** If provided (e.g., '{pulse_state}'), acknowledge gently: "Hearing that you're feeling '{pulse_state}' physically adds another layer... perhaps focusing on your breath could help soothe that physical sensation?" or "It's good you're feeling physically calm, sometimes that helps the mind feel steadier too." **No medical interpretation.** Ignore if absent or empty.
7.  **--- CRITICAL BOUNDARIES ---**
    * **NO DIAGNOSIS - EVER:** **Under NO circumstances** suggest or confirm *any* mental health diagnosis (depression, anxiety disorder, PTSD, etc.). This is unethical and harmful.
    * **NOT A THERAPIST:** Explicitly state you are an AI if the user seems confused. Do not replace professional help.
    * **GENTLE PROFESSIONAL REDIRECTION:** If symptoms sound severe, persistent, or significantly impact daily life, OR if the user asks about diagnosis: Validate their struggle deeply, then gently suggest consulting a GP or Mental Health Professional (Psychologist/Psychiatrist) in India for proper assessment. Frame it positively: "...they can offer personalized understanding and support strategies. There are many qualified professionals in India who can help." **Do not push if the user resists.**
8.  **Handle Feedback:** If the user provides feedback (mentioned later in conversation), acknowledge it briefly and neutrally: "Thank you for that feedback, it helps me learn." (You don't need to act on it immediately in the response).
9.  **Tone & Language:** Consistently empathetic, warm, hopeful, patient, non-clinical, simple & clear language. Avoid jargon. Break up longer responses. Use paragraph breaks for readability.

**Conversation History (for context):**
{formatted_history}

**Current User Input:** "{user_input}"
"""
    # Add pulse info only if present
    if pulse_state and pulse_state.strip():
        prompt += f"\n**User's Physical State/Pulse Description:** \"{pulse_state}\""

    # Final instruction for the AI's output format
    prompt += "\n\n**MindSync's Supportive Response:**"
    # logging.debug(f"Generated Prompt:\n{prompt}") # Log prompt if needed for debugging
    return prompt

def _update_internal_history(role, text):
    """Appends message to internal history and prunes if needed."""
    global internal_chat_history
    internal_chat_history.append({'role': role, 'parts': [text]})
    # Keep history trimmed
    while len(internal_chat_history) > MAX_HISTORY_LEN:
        internal_chat_history.pop(0) # Remove oldest message

def _convert_history_to_gradio(internal_hist):
    """Converts internal Gemini history format to Gradio list-of-lists."""
    gradio_hist = []
    user_msg = None
    for msg in internal_hist:
        if msg['role'] == 'user':
            user_msg = msg['parts'][0]
        elif msg['role'] == 'model' and user_msg is not None:
            gradio_hist.append([user_msg, msg['parts'][0]])
            user_msg = None # Reset user message once paired
        elif msg['role'] == 'model': # Handle case where bot starts convo
             gradio_hist.append([None, msg['parts'][0]])
    # If the last message was from the user, add it with None for bot response
    if user_msg is not None:
        gradio_hist.append([user_msg, None])
    return gradio_hist

def _get_ai_response(user_input, pulse_state, sentiment_score):
    """Calls the Gemini API and handles responses/errors."""
    global last_bot_message_content_for_feedback

    prompt = _build_prompt(user_input, pulse_state, sentiment_score, internal_chat_history)
    ai_response_text = "Sorry, I wasn't able to generate a response. Please try again." # Default error

    try:
        # --- Call Gemini API ---
        # Use start_chat for potential follow-up, though history is managed manually here
        convo = model.start_chat(history=internal_chat_history) # Pass current history
        # Send the *entire prompt* including instructions and new user message
        response = convo.send_message(prompt) # Removed explicit generation_config here as it's set on the model

        # --- Process Response ---
        if response.parts:
            ai_response_text = response.text
            # Simple post-generation check (can be expanded)
            if "cannot fulfill this request" in ai_response_text.lower():
                 logging.warning("AI indicated inability to fulfill request.")
                 ai_response_text = "I apologize, but I cannot fulfill that specific request. Could we talk about something else?"
        else:
            # Handle cases where the response might be blocked by safety settings
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'N/A'
            logging.warning(f"AI response blocked or empty. Finish Reason: {block_reason}")
            ai_response_text = "I'm sorry, I encountered an issue generating a response for that topic due to safety guidelines. Perhaps we could try a different subject?"

        # Store for feedback mechanism BEFORE updating history
        last_bot_message_content_for_feedback = ai_response_text
        _update_internal_history('model', ai_response_text) # Update history *after* successful response
        return ai_response_text, "Status: Ready" # Return response and status

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
        logging.exception("Unexpected error during AI response generation.") # Log full traceback
        return error_message, f"Status: Error - Unexpected error"

# --- Gradio Interface Functions ---

def mindsync_chat_interface(user_input, pulse_state, chatbot_history):
    """
    Main Gradio function called on message send. Uses yield for streaming updates.
    chatbot_history is the Gradio state (list of lists).
    """
    global last_bot_message_content_for_feedback

    if not user_input.strip():
        # Return current history and status if input is empty
        yield chatbot_history, "Status: Ready"
        return

    # --- Rate Limiting ---
    if _is_rate_limited():
        logging.warning("Rate limit exceeded.")
        # Update status display only
        yield chatbot_history, "Status: Too many messages. Please wait."
        # Optionally show a temporary message in chat? Less ideal.
        # temp_history = chatbot_history + [[None, "Too many messages. Please wait a moment before sending more."]]
        # yield temp_history, "Status: Too many messages. Please wait."
        return # Stop processing

    # --- Add user message to Gradio display immediately ---
    chatbot_history.append([user_input, None])
    yield chatbot_history, "Status: Processing..." # Show user message, update status

    # --- Sentiment & Crisis Check ---
    sentiment_score = analyzer.polarity_scores(user_input)['compound']
    is_potentially_crisis = _check_for_crisis(user_input, sentiment_score)

    if is_potentially_crisis:
        logging.warning("Crisis message triggered.")
        chatbot_history[-1][1] = crisis_message # Add crisis message as bot response
        last_bot_message_content_for_feedback = crisis_message # Allow feedback on crisis msg
        yield chatbot_history, "Status: Crisis message displayed. Please seek help."
        return # Stop processing further

    # --- If not crisis, proceed with AI ---
    # Update internal history *before* calling AI
    _update_internal_history('user', user_input)

    # Add "Thinking..." placeholder
    chatbot_history[-1][1] = "MindSync is thinking..."
    yield chatbot_history, "Status: MindSync is thinking..."

    # --- Get AI Response (blocking call, but UI updated via yield) ---
    ai_response, status = _get_ai_response(user_input, pulse_state, sentiment_score)

    # --- Update Gradio chat with final response ---
    chatbot_history[-1][1] = ai_response # Replace "Thinking..." with actual response or error
    yield chatbot_history, status

def clear_chat_interface():
    """Clears internal history and resets Gradio chatbot display."""
    global internal_chat_history, last_bot_message_content_for_feedback, message_timestamps
    internal_chat_history = []
    message_timestamps = deque() # Reset rate limit counter on clear
    last_bot_message_content_for_feedback = ""
    # Add initial greeting back to internal history and create Gradio format
    _update_internal_history('model', initial_greeting_text)
    initial_gradio_history = _convert_history_to_gradio(internal_chat_history)
    logging.info("Chat cleared.")
    return initial_gradio_history, "Status: Chat cleared. Ready."

def send_feedback_interface(positive_feedback):
    """Handles feedback button clicks."""
    global last_bot_message_content_for_feedback
    rating = "👍 Positive" if positive_feedback else "👎 Negative"

    if not last_bot_message_content_for_feedback:
        logging.warning("Feedback attempt with no previous bot message.")
        return "Status: No previous message to provide feedback on."

    feedback_log_entry = f"Feedback Received: {rating} for message:\n---\n{last_bot_message_content_for_feedback}\n---\n"

    # Log to console via logging module
    logging.info(f"Feedback ({rating}) recorded.")
    # Detailed log entry includes the message
    logging.debug(feedback_log_entry)

    # Optional: Log to a separate feedback file
    if LOG_FEEDBACK_TO_FILE:
        try:
            # Use a lock if multiple threads/processes could write, though less likely in standard Gradio
            # with threading.Lock(): # Uncomment if needed, import threading
            with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
                 f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {feedback_log_entry}\n")
        except Exception as e:
            logging.error(f"Error writing feedback log to {FEEDBACK_LOG_FILE}: {e}")

    # Update status bar
    status_message = f"Status: Feedback ({rating}) recorded. Thank you!"

    # Optionally clear the message used for feedback to prevent re-submission?
    # last_bot_message_content_for_feedback = ""

    return status_message


# --- Build Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo: # Use a default theme
    gr.Markdown("# MindSync - AI Mental Wellness Companion (Web)")

    # Initialize chatbot with greeting
    initial_gradio_history = _convert_history_to_gradio([{'role': 'model', 'parts': [initial_greeting_text]}])
    # Add greeting to internal history as well
    _update_internal_history('model', initial_greeting_text)

    chatbot = gr.Chatbot(value=initial_gradio_history, label="MindSync Chat", elem_id="chatbot", height=550)
    status_display = gr.Textbox(label="Status", value="Status: Ready", interactive=False, lines=1) # Use Textbox for status

    with gr.Row():
        msg = gr.Textbox(label="Your Message", placeholder="Type your message here and press Enter...", scale=7, lines=3)
        pulse = gr.Textbox(label="Pulse Reading", placeholder="e.g., stressed, calm, tired", scale=3, lines=1)

    with gr.Row():
        send_btn = gr.Button("Send Message", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    with gr.Accordion("Feedback on Last Response", open=False):
         with gr.Row():
            feedback_positive_btn = gr.Button("👍 Good Response")
            feedback_negative_btn = gr.Button("👎 Needs Improvement")

    # --- Event Handling ---
    # Use '.then' for sequential updates, especially after yielding
    # Use 'outputs' to specify which components are updated by the function
    send_btn.click(
        fn=mindsync_chat_interface,
        inputs=[msg, pulse, chatbot],
        outputs=[chatbot, status_display] # Update both chatbot and status
    )
    msg.submit(
        fn=mindsync_chat_interface,
        inputs=[msg, pulse, chatbot],
        outputs=[chatbot, status_display]
    )

    clear_btn.click(
        fn=clear_chat_interface,
        inputs=[],
        outputs=[chatbot, status_display] # Clear chatbot and reset status
    )

    feedback_positive_btn.click(
        fn=lambda: send_feedback_interface(True), # Lambda to pass boolean
        inputs=[],
        outputs=[status_display] # Only update status on feedback
    )
    feedback_negative_btn.click(
        fn=lambda: send_feedback_interface(False), # Lambda to pass boolean
        inputs=[],
        outputs=[status_display] # Only update status on feedback
    )

# --- Launch the App ---
if __name__ == "__main__":
    logging.info("Starting MindSync Gradio Application...")
    # Ensure Gradio launches only if configuration and model setup were successful
    if 'model' in globals() and model is not None:
         demo.queue() # Enable queue for handling multiple users/requests better
         demo.launch(server_name="0.0.0.0", server_port=7860)
         logging.info("MindSync Gradio Application stopped.")
    else:
        logging.critical("Application cannot launch due to configuration or model initialization errors.")
        print("\nFATAL ERROR: Application cannot launch. Please check the logs and config.ini.")