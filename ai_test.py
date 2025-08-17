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
# Suppress warnings for Wav2Vec2
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
    st.error("Email credentials not found. Please set 'MAIL_SENDER_EMAIL' and 'MAIL_SENDER_PASS' in secrets.toml.")
    MAIL_SENDER_EMAIL = None
    MAIL_SENDER_PASS = None

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Audio & Text Summarizer", layout="wide")
st.title("🎧 Audio & Text Summarizer")
st.write("Upload an audio file for transcription or paste your text. Click 'Generate Summary' to summarize automatically.")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    audio_transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    return summarizer, audio_transcriber

summarizer, audio_transcriber = load_models()

# -----------------------------
# Helper Functions
# -----------------------------
def log_action(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def save_temp_audio(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path, temp_dir

def transcribe_audio(audio_path):
    if not audio_transcriber:
        return "[ERROR: Audio transcriber not loaded]"
    try:
        log_action(f"Transcribing: {audio_path}")
        result = audio_transcriber(audio_path, chunk_length_s=30, stride_length_s=[4,2])
        return result["text"]
    except Exception as e:
        return f"[ERROR: Transcription failed] {e}"

def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    tokens = 0
    for w in words:
        tokens += 1
        current_chunk.append(w)
        if tokens >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            tokens = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(text):
    if not summarizer:
        return "[ERROR: Summarization model not loaded]"
    try:
        summary_chunks = []
        for chunk in chunk_text(text):
            result = summarizer(chunk, max_length=130, min_length=30, do_sample=True, temperature=0.7, top_p=0.9)
            summary_chunks.append(result[0]["summary_text"])
        combined_summary = " ".join(summary_chunks)
        # format as bullet points automatically
        sentences = re.split(r'(?<=[.!?]) +', combined_summary)
        combined_summary = "\n".join([f"• {s.strip()}" for s in sentences if len(s.strip()) > 20])
        return combined_summary
    except Exception as e:
        return f"[ERROR: Summarization failed] {e}"

def send_email(recipient, subject, body):
    if not MAIL_SENDER_EMAIL or not MAIL_SENDER_PASS:
        st.error("Email not configured.")
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
        log_action("Email sent successfully.")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def cleanup_temp_dir(temp_dir):
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

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
# Upload / Paste Input
# -----------------------------
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3","wav","m4a","ogg"])
text_input = st.text_area("Or Paste Text Here", height=150)

if uploaded_file:
    cleanup_temp_dir(st.session_state.audio_temp_dir)
    temp_file_path, temp_dir = save_temp_audio(uploaded_file)
    st.session_state.audio_temp_dir = temp_dir
    st.audio(temp_file_path)
    st.session_state.transcription = transcribe_audio(temp_file_path)

elif text_input.strip():
    st.session_state.transcription = text_input

# -----------------------------
# Generate Summary
# -----------------------------
if st.session_state.transcription:
    st.subheader("📝 Transcription / Text Input")
    st.text_area("Full Text", st.session_state.transcription, height=250)

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            st.session_state.summary = summarize_text(st.session_state.transcription)

# -----------------------------
# Display Summary & Email
# -----------------------------
if st.session_state.summary:
    st.subheader("📌 Summary")
    st.text_area("Summary", st.session_state.summary, height=250)

    st.subheader("✉️ Send Summary via Email")
    with st.form(key="email_form"):
        recipient_email = st.text_input("Recipient Email", value="kulurunagavardhanreddy@gmail.com")
        email_subject = st.text_input("Email Subject", value="Your Summary")
        if st.form_submit_button("Send Email"):
            if send_email(recipient_email, email_subject, st.session_state.summary):
                st.success("Email sent successfully!")

# -----------------------------
# Cleanup temp directory
# -----------------------------
if st.session_state.audio_temp_dir:
    cleanup_temp_dir(st.session_state.audio_temp_dir)
    st.session_state.audio_temp_dir = None
