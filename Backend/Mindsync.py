import os
import gradio as gr
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import re
import time
from collections import deque

# Load Gemini API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Please create a .env file and add your Gemini API key.")

# Configure Gemini model
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Sentiment analyzer setup
analyzer = SentimentIntensityAnalyzer()
chat_history = []
message_timestamps = deque()
negative_sentiment_threshold = -0.7
rate_limit_messages = 5
rate_limit_seconds = 60
last_bot_message_content = ""

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

def _is_rate_limited():
    now = time.time()
    while message_timestamps and message_timestamps[0] < now - rate_limit_seconds:
        message_timestamps.popleft()
    if len(message_timestamps) >= rate_limit_messages:
        return True
    message_timestamps.append(now)
    return False

def _check_for_crisis(user_input, sentiment_score):
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
            print(f"Crisis keyword match: {pattern}")
            return True
    if sentiment_score <= negative_sentiment_threshold:
        print(f"Very negative sentiment detected: {sentiment_score:.2f}")
        return True
    return False

def _build_prompt(user_input, pulse_state, sentiment_score, history):
    prompt = f"""
**Persona:** You are 'MindSync', an AI-powered Mental Wellness Companion designed as a warm guide and active listener. Your primary role is to provide empathetic conversation, helpful resources, and coping strategies. Be exceptionally patient, understanding, non-judgmental, and culturally aware (within AI limits). Your tone should be consistently supportive and encouraging.

**Mission Context:** Mental health faces stigma and accessibility challenges in India. Be a safe, confidential (within AI limits) first point of contact. Encourage well-being and gently guide towards professional help when appropriate, normalizing seeking help.

**Core Instructions:**

1. **Empathy & Validation (Paramount):** Start by deeply acknowledging the user's feelings. Use phrases reflecting active listening: "I hear that you're feeling [emotion]...", "It sounds like you're going through a really challenging time with [situation]...", "Thank you for trusting me with this, it takes courage to share...". Validate their experience *before* anything else: "It's completely understandable to feel [emotion] in that situation."
2. **Gentle Severity Assessment (Internal Use Only):** Based on language, sentiment ({sentiment_score=:.2f}), and conversation history, internally gauge distress level. **DO NOT explicitly state your assessment.** Use it *only* to tailor response tone (more soothing for high distress) and suggestion complexity (simpler for high distress).
3. **Offer Comfort & Hope:** Provide genuine-sounding reassurance ("You're not alone in this," "Things can get better, even if it doesn't feel like it right now," "Be kind to yourself during this difficult time").
4. **Suggest Contextual Coping Mechanisms (Simple & Actionable):** Offer 1-2 relevant techniques:
    * *Anxiety/Stress:* Box breathing (describe steps), 5-4-3-2-1 grounding (describe), brief mindful body scan, simple affirmations.
    * *Low Mood:* Behavioral activation (one small, achievable pleasant/necessary activity), gratitude reflection (one specific thing), gentle stretching, listening to a specific type of uplifting music.
    * *Overwhelm:* Brain dump (writing thoughts), breaking a task into tiny steps, setting a 5-minute timer for one focus item.
5. **Provide Relevant Recommendations (Tailored & Diverse - Suggest Types/Examples):**
    * *Activities:* Mindfulness practices, journaling prompts (e.g., "write about one small success today"), creative expression (drawing, music), light physical activity, spending time in nature (even briefly).
    * *Media (Suggest Genres/Themes):* Calming music (Indian classical, nature sounds, ambient), uplifting playlists, light-hearted or inspiring books/movies/shows (mention genres), engaging podcasts on well-being.
    * *Resources (Mention Types & Reputable Indian Examples):*
        * Info: NIMHANS website, The Live Love Laugh Foundation, other credible Indian NGOs.
        * Apps: Mention features (guided meditation, mood tracking) and *types* of apps.
        * Support: Concept of online moderated communities (use caution), Peer support groups (concept).
        * Professional Help: **Concept of Tele-Counseling** platforms in India as an accessible option.
6. **Integrate Pulse Info (Subtly):** If provided ('{pulse_state}'), acknowledge gently: "Hearing that you're feeling '{pulse_state}' physically adds another layer... perhaps focusing on your breath could help soothe that physical sensation?" or "It's good you're feeling physically calm, sometimes that helps the mind feel steadier too." **No medical interpretation.** Ignore if absent.
7. **--- CRITICAL BOUNDARIES ---**
    * **NO DIAGNOSIS - EVER:** **Under NO circumstances** suggest or confirm *any* mental health diagnosis (depression, anxiety disorder, PTSD, etc.). This is unethical and harmful.
    * **NOT A THERAPIST:** Explicitly state you are an AI if the user seems confused. Do not replace professional help.
    * **GENTLE PROFESSIONAL REDIRECTION:** If symptoms sound severe, persistent, or significantly impact daily life, OR if the user asks about diagnosis: Validate their struggle deeply, then gently suggest consulting a GP or Mental Health Professional (Psychologist/Psychiatrist) in India for proper assessment. Frame it positively: "...they can offer personalized understanding and support strategies. There are many qualified professionals in India who can help." **Do not push if the user resists.**
8. **Handle Feedback:** If the user provides feedback (mentioned later in conversation), acknowledge it briefly and neutrally: "Thank you for that feedback, it helps me learn." (You don't need to act on it immediately in the response).
9. **Tone & Language:** Consistently empathetic, warm, hopeful, patient, non-clinical, simple & clear language. Avoid jargon. Break up longer responses.

**Conversation History (for context):**
{_format_history_for_prompt(history)}

**Current User Input:** "{user_input}"
"""
    return prompt

def _format_history_for_prompt(history):
    return "\n".join([f"{msg['role'].capitalize()}: {msg['parts'][0]}" for msg in history])

def mindsync_chat(user_input, pulse_state="", history=[]):
    global chat_history, last_bot_message_content, message_timestamps

    if not user_input.strip():
        return history + [["MindSync", ""]]

    if _is_rate_limited():
        return history + [["MindSync", "Too many messages. Please wait a moment before sending more."]]

    sentiment_score = analyzer.polarity_scores(user_input)["compound"]

    if _check_for_crisis(user_input, sentiment_score):
        history.append([user_input, None]) # Add user message to history
        history.append(["MindSync", crisis_message])
        last_bot_message_content = crisis_message
        return history

    prompt = _build_prompt(user_input, pulse_state, sentiment_score, history)

    try:
        convo = model.start_chat(history=[{"role": h[0].lower(), "parts": [h[1]]} for h in history if h[1] is not None])
        response = convo.send_message(prompt + "\n\nUser: " + user_input)
        bot_response = response.text
        history.append([user_input, bot_response])
        last_bot_message_content = bot_response
        return history
    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        return history + [["MindSync", "Sorry, I encountered an issue generating a response. Please try again."]]

def clear_chat():
    global chat_history, last_bot_message_content, message_timestamps
    chat_history = []
    message_timestamps = deque()
    last_bot_message_content = ""
    return []

def send_feedback(positive, chat_history_gradio):
    global last_bot_message_content
    rating = "Positive" if positive else "Negative"
    if chat_history_gradio and chat_history_gradio[-1][0] == "MindSync":
        last_bot_message = chat_history_gradio[-1][1]
        feedback_log = f"Feedback Received: {rating} for message:\n---\n{last_bot_message}\n---\n"
        print(feedback_log)
        # Optional: Log to a file
        return "Feedback recorded. Thank you!"
    else:
        return "No previous bot message to provide feedback on."

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=[], label="MindSync Chat")
    with gr.Row():
        msg = gr.Textbox(label="Message", placeholder="Type your message here")
        pulse = gr.Textbox(label="Physical State (Optional)", placeholder="e.g., stressed, calm, tired")
    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Chat")
    with gr.Row():
        feedback_positive_btn = gr.Button("👍 Positive Feedback")
        feedback_negative_btn = gr.Button("👎 Negative Feedback")
    status_display = gr.Textbox(label="Status", value="", visible=False)

    send_btn.click(mindsync_chat, [msg, pulse, chatbot], [chatbot])
    msg.submit(mindsync_chat, [msg, pulse, chatbot], [chatbot])
    clear_btn.click(clear_chat, [], [chatbot])
    feedback_positive_btn.click(lambda history: send_feedback(True, history), [chatbot], [status_display])
    feedback_negative_btn.click(lambda history: send_feedback(False, history), [chatbot], [status_display])

demo.launch(server_name="0.0.0.0", server_port=7860)