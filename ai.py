import streamlit as st
import os
import tempfile
from transformers import pipeline
from pydub import AudioSegment
import whisper
import re
import smtplib
from email.message import EmailMessage
from datetime import datetime
import warnings

# Suppress the specific Whisper FP16 warning which is harmless on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

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
        # Load the summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # Load the Whisper model
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
    try:
        # Use tempfile to create a secure temporary directory and file
        temp_dir = tempfile.mkdtemp()
        original_temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(original_temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert the audio to WAV using pydub
        sound = AudioSegment.from_file(original_temp_path)
        wav_path = os.path.join(temp_dir, f"{os.path.basename(uploaded_file.name).split('.')[0]}.wav")
        sound.export(wav_path, format="wav")
        
        # Clean up the original uploaded file
        os.remove(original_temp_path)
        
        return wav_path, temp_dir
    except Exception as e:
        st.error(f"Error converting audio file. Make sure it's a valid audio format: {e}")
        return None, None

def transcribe_audio(audio_path):
    """Transcribes audio using the Whisper model."""
    if not whisper_model:
        return "[ERROR: Whisper model not loaded]"
    try:
        log_action(f"Starting transcription for {audio_path}")
        result = whisper_model.transcribe(audio_path)
        log_action("Transcription complete.")
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        log_action(f"Transcription failed: {e}")
        return "[ERROR: Transcription failed]"

def chunk_text(text, max_tokens=800):
    """Breaks down long text into smaller chunks for summarization."""
    words = text.split()
    chunks = []
    current_chunk_words = []
    tokens = 0
    for w in words:
        tokens += 1
        current_chunk_words.append(w)
        if tokens >= max_tokens:
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words = []
            tokens = 0
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
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
            # Filter out very short sentences that are likely just artifacts
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
if "audio_temp_dir" not in st.session_state:
    st.session_state.audio_temp_dir = None

# -----------------------------
# Cleanup function
# -----------------------------
def cleanup_temp_files():
    """Removes temporary files after use."""
    if st.session_state.audio_temp_dir and os.path.exists(st.session_state.audio_temp_dir):
        try:
            for file_name in os.listdir(st.session_state.audio_temp_dir):
                file_path = os.path.join(st.session_state.audio_temp_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(st.session_state.audio_temp_dir)
            log_action(f"Cleaned up temporary directory: {st.session_state.audio_temp_dir}")
        except Exception as e:
            st.error(f"Error during cleanup: {e}")
    st.session_state.audio_temp_dir = None

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
    # Always clean up previous files before processing a new one
    cleanup_temp_files()
    
    temp_file_path, temp_dir = save_temp_audio(uploaded_file)
    if temp_file_path and temp_dir:
        st.session_state.audio_temp_dir = temp_dir
        st.audio(temp_file_path)
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                st.session_state.transcription = transcribe_audio(temp_file_path)
                st.session_state.summary = ""
                log_action("Audio has been transcribed.")
                
        # To make sure temp files are cleaned up on app reruns
        st.session_state.temp_file_path = temp_file_path
        st.session_state.temp_dir = temp_dir


# -----------------------------
# Process Text Input
# -----------------------------
if text_input.strip() != "":
    log_action("User provided text input.")
    # Clean up any residual audio files when switching to text input
    cleanup_temp_files()
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

# Ensure cleanup happens when the session ends or app is re-run with a new file
if st.session_state.audio_temp_dir:
    import shutil
    shutil.rmtree(st.session_state.audio_temp_dir, ignore_errors=True)
    st.session_state.audio_temp_dir = None
