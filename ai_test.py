import streamlit as st
import os
import tempfile
from transformers import pipeline
import re
import smtplib
from email.message import EmailMessage
from datetime import datetime
import shutil
import warnings

# -----------------------------
# Suppress specific warnings
# -----------------------------
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at facebook/wav2vec2-base-960h were not used")
warnings.filterwarnings("ignore", message="You are using a model of type wav2vec2_ctc to automatically transcribe audio")

# -----------------------------
# Load email credentials
# -----------------------------
try:
    MAIL_SENDER_EMAIL = st.secrets["MAIL_SENDER_EMAIL"]
    MAIL_SENDER_PASS = st.secrets.get("MAIL_SENDER_PASS")
except KeyError:
    st.error("Email credentials not found. Please set them in secrets.toml.")
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
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        audio_transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
        return summarizer, audio_transcriber
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

summarizer, audio_transcriber = load_models()

# -----------------------------
# Helper Functions
# -----------------------------
def log_action(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def save_temp_audio(uploaded_file):
    try:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path, temp_dir
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return None, None

def transcribe_audio(audio_path):
    if not audio_transcriber:
        return "[ERROR: Audio transcriber model not loaded]"
    try:
        log_action(f"Transcribing audio: {audio_path}")
        result = audio_transcriber(audio_path, chunk_length_s=30, stride_length_s=[4, 2])
        log_action("Transcription complete.")
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return "[ERROR: Transcription failed]"

def chunk_text(text, max_tokens=800):
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

def summarize_text(text, max_length=None, min_length=10, bullet_style=True):
    """Generates a professional summary with no hard limits."""
    if not summarizer:
        return "[ERROR: Summarization model not loaded]"
    try:
        words_count = len(text.split())
        # Auto-adjust max length if not specified
        if not max_length:
            max_length = max(50, min(words_count // 2, 500))

        summary_chunks = []
        for chunk in chunk_text(text, max_tokens=500):
            if not chunk.strip():
                continue
            summary_list = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            if summary_list and "summary_text" in summary_list[0]:
                summary_chunks.append(summary_list[0]["summary_text"])

        if not summary_chunks:
            return "[ERROR: No summary could be generated]"

        combined_summary = " ".join(summary_chunks)
        if bullet_style:
            sentences = re.split(r'(?<=[.!?]) +', combined_summary)
            combined_summary = "\n".join([f"• {s.strip()}" for s in sentences if len(s.strip()) > 20])

        return combined_summary
    except Exception as e:
        return f"[ERROR: Summarization failed] {e}"

def send_email(recipient, subject, body):
    if not MAIL_SENDER_EMAIL or not MAIL_SENDER_PASS:
        st.error("Email is not configured.")
        return False
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = MAIL_SENDER_EMAIL
        msg["To"] = recipient

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(MAIL_SENDER_EMAIL, MAIL_SENDER_PASS)
            smtp.send_message(msg)
        log_action("Email sent successfully.")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        log_action(f"Email failed: {e}")
        return False

def cleanup_temp_files():
    if st.session_state.audio_temp_dir and os.path.exists(st.session_state.audio_temp_dir):
        try:
            shutil.rmtree(st.session_state.audio_temp_dir, ignore_errors=True)
            log_action(f"Removed temp dir: {st.session_state.audio_temp_dir}")
        except Exception as e:
            st.error(f"Cleanup error: {e}")
    st.session_state.audio_temp_dir = None

# -----------------------------
# Session State
# -----------------------------
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "audio_temp_dir" not in st.session_state:
    st.session_state.audio_temp_dir = None

# -----------------------------
# Upload Audio / Text
# -----------------------------
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a", "ogg"])
text_input = st.text_area("Or Paste Text Here", height=150)

if uploaded_file:
    cleanup_temp_files()
    temp_file_path, temp_dir = save_temp_audio(uploaded_file)
    if temp_file_path and temp_dir:
        st.session_state.audio_temp_dir = temp_dir
        st.audio(temp_file_path)
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                st.session_state.transcription = transcribe_audio(temp_file_path)
                st.session_state.summary = ""

if text_input.strip():
    cleanup_temp_files()
    st.session_state.transcription = text_input
    st.session_state.summary = ""

# -----------------------------
# Display Transcription
# -----------------------------
if st.session_state.transcription:
    st.subheader("📝 Transcription / Text Input")
    st.text_area("Full Text", st.session_state.transcription, height=250, key="full_text_area")

    words_count = len(st.session_state.transcription.split())
    max_len = st.slider("Max Summary Length", min_value=10, max_value=max(50, words_count), value=min(130, words_count))
    min_len = st.slider("Min Summary Length", min_value=5, max_value=max_len, value=min(30, max_len))
    bullet_mode = st.checkbox("Format Summary in Bullet Points", value=True)

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            st.session_state.summary = summarize_text(
                st.session_state.transcription,
                max_length=max_len,
                min_length=min_len,
                bullet_style=bullet_mode
            )
            log_action("Summary generated.")

# -----------------------------
# Display Summary & Email
# -----------------------------
if st.session_state.summary:
    st.subheader("📌 Summary")
    st.text_area("Bullet Point Summary", st.session_state.summary, height=250, key="summary_text_area")

    st.subheader("✉️ Send Summary via Email")
    with st.form(key='email_form'):
        recipient_email = st.text_input("Recipient Email", value="kulurunagavardhanreddy@gmail.com")
        email_subject = st.text_input("Email Subject", value="Your Summary")
        submit_button = st.form_submit_button("Send Email")

        if submit_button:
            if send_email(recipient_email, email_subject, st.session_state.summary):
                st.success("Email sent successfully!")

# -----------------------------
# Cleanup
# -----------------------------
if st.session_state.audio_temp_dir:
    shutil.rmtree(st.session_state.audio_temp_dir, ignore_errors=True)
    st.session_state.audio_temp_dir = None
