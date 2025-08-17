import streamlit as st
import os
from transformers import pipeline
from pydub import AudioSegment
import whisper
import re
import smtplib
from email.message import EmailMessage
from datetime import datetime

# -----------------------------
# Load email credentials from secrets.toml
# -----------------------------
try:
    MAIL_SENDER_EMAIL = st.secrets["MAIL_SENDER_EMAIL"]
    MAIL_SENDER_PASS = st.secrets["MAIL_SENDER_PASS"]
except KeyError:
    st.error("Email credentials not found. Please set 'MAIL_SENDER_EMAIL' and 'MAIL_SENDER_PASS' in secrets.toml.")
    MAIL_SENDER_EMAIL = None
    MAIL_SENDER_PASS = None

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Audio & Text Summarizer", layout="wide")
st.title("🎧 Audio & Text Summarizer")
st.write("Upload an audio file for transcription or paste your text directly to summarize.")

# -----------------------------
# Load Models Once
# -----------------------------
@st.cache_resource
def load_models():
    """Loads the summarization and whisper models and caches them."""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        whisper_model = whisper.load_model("base")
        return summarizer, whisper_model
    except Exception as e:
        st.error(f"Error loading models. Please check your dependencies: {e}")
        return None, None

summarizer, whisper_model = load_models()

# -----------------------------
# Helper Functions
# -----------------------------
def log_action(message):
    """Logs a message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def save_temp_audio(uploaded_file):
    """Saves the uploaded audio file to a temporary location and converts it to WAV."""
    temp_file = f"./temp_{uploaded_file.name}"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        sound = AudioSegment.from_file(temp_file)
        wav_path = f"./temp_{uploaded_file.name}.wav"
        sound.export(wav_path, format="wav")
        os.remove(temp_file)
        return wav_path
    except Exception as e:
        st.error(f"Error converting audio file: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribes audio using the Whisper model."""
    if not whisper_model:
        return "[ERROR: Whisper model not loaded]"
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return "[ERROR: Transcription failed]"

def chunk_text(text, max_tokens=800):
    """Breaks down long text into smaller chunks for summarization."""
    words = text.split()
    chunks = []
    current = []
    tokens = 0
    for w in words:
        tokens += 1
        current.append(w)
        if tokens >= max_tokens:
            chunks.append(" ".join(current))
            current = []
            tokens = 0
    if current:
        chunks.append(" ".join(current))
    return chunks

def summarize_text(text, max_length=130, min_length=30, bullet_style=True):
    """
    Generates a summary of the provided text.
    Uses do_sample=True to generate a different summary each time.
    """
    if not summarizer:
        return "[ERROR: Summarization model not loaded]"
    try:
        summary_chunks = []
        text_chunks = chunk_text(text, max_tokens=800)
        for chunk in text_chunks:
            # The key change is here: do_sample=True
            summary_list = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=True, # This enables sampling for different outputs
                temperature=0.7, # Controls the randomness (0.0 to 1.0)
                top_p=0.9 # Controls the diversity of the output
            )
            summary_chunks.append(summary_list[0]["summary_text"])
        combined_summary = " ".join(summary_chunks)

        if bullet_style:
            sentences = re.split(r'(?<=[.!?]) +', combined_summary)
            combined_summary = "\n".join([f"• {s.strip()}" for s in sentences if len(s.strip()) > 20])
        return combined_summary
    except Exception as e:
        return f"[ERROR: Summarization failed] {e}"

def send_email(recipient, subject, body):
    """Sends an email with the summary."""
    if not MAIL_SENDER_EMAIL or not MAIL_SENDER_PASS:
        st.error("Email functionality is not configured.")
        return False
        
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = MAIL_SENDER_EMAIL
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(MAIL_SENDER_EMAIL, MAIL_SENDER_PASS)
            smtp.send_message(msg)
        log_action("Summary email sent successfully.")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        log_action(f"Failed to send email: {e}")
        return False

# -----------------------------
# Session State
# -----------------------------
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# -----------------------------
# Upload Audio / Paste Text
# -----------------------------
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a", "ogg"])
text_input = st.text_area("Or Paste Text Here", height=150)

# -----------------------------
# Process Audio
# -----------------------------
if uploaded_file:
    log_action("User uploaded an audio file.")
    temp_file_path = save_temp_audio(uploaded_file)
    if temp_file_path:
        st.audio(temp_file_path)
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                st.session_state.transcription = transcribe_audio(temp_file_path)
                st.session_state.summary = ""
                log_action("Audio has been transcribed.")

# -----------------------------
# Process Text Input
# -----------------------------
if text_input.strip() != "":
    log_action("User provided text input.")
    st.session_state.transcription = text_input
    st.session_state.summary = ""

# -----------------------------
# Display Transcription / Text
# -----------------------------
if st.session_state.transcription:
    st.subheader("📝 Transcription / Text Input")
    st.text_area("Full Text", st.session_state.transcription, height=250, key="full_text_area")

    # --- Only now show summary options ---
    max_len = st.slider("Max Summary Length", min_value=50, max_value=500, value=130)
    min_len = st.slider("Min Summary Length", min_value=10, max_value=200, value=30)
    bullet_mode = st.checkbox("Format Summary in Bullet Points", value=True)

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            st.session_state.summary = summarize_text(
                st.session_state.transcription,
                max_length=max_len,
                min_length=min_len,
                bullet_style=bullet_mode
            )
            log_action("Summary has been generated.")

# -----------------------------
# Display Summary
# -----------------------------
if st.session_state.summary:
    st.subheader("📌 Summary")
    st.text_area("Bullet Point Summary", st.session_state.summary, height=250, key="summary_text_area")

    # --- Email sending feature using a form to prevent reload ---
    st.subheader("✉️ Send Summary via Email")
    # Wrap email inputs and button in a form
    with st.form(key='email_form'):
        recipient_email = st.text_input("Recipient Email", value="kulurunagavardhanreddy@gmail.com", key="recipient_email")
        email_subject = st.text_input("Email Subject", value="Your Summary", key="email_subject")
        submit_button = st.form_submit_button("Send Email")

        if submit_button:
            if send_email(recipient_email, email_subject, st.session_state.summary):
                st.success("Email sent successfully!")

