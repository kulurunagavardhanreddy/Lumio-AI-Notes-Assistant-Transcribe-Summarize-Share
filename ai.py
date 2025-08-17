import streamlit as st
import whisper
from transformers import pipeline
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import tempfile
import pathlib

# Set up page config
st.set_page_config(
    page_title="AI Notes Summarizer",
    page_icon="📝",
    layout="centered",
)

# -------------------------
# Email credentials from Streamlit secrets
# -------------------------
def load_mail_credentials():
    """
    Loads email credentials from Streamlit's secrets management.
    Handles the case where secrets are not configured.
    """
    try:
        sender_email = st.secrets["mail"]["MAIL_SENDER_EMAIL"]
        sender_password = st.secrets["mail"]["MAIL_SENDER_PASS"]
        return sender_email, sender_password
    except KeyError as e:
        # Provide a user-friendly message for a specific error.
        st.error(f"Error: Missing key in Streamlit secrets. "
                 f"Please add `MAIL_SENDER_EMAIL` and `MAIL_SENDER_PASS` to `secrets.toml`.")
        st.error(f"Details: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading mail credentials: {e}")
        return None, None

# -------------------------
# Helper functions with caching
# -------------------------
@st.cache_resource(show_spinner="Loading Whisper model, this may take a moment...")
def load_whisper_model(model_name="base.en"): # Use a smaller model for faster loading
    """
    Loads the Whisper model and caches it using st.cache_resource.
    This ensures the model is downloaded and loaded only once per deployment.
    """
    try:
        model = whisper.load_model(model_name)
        return model
    except Exception as e:
        st.error(f"Whisper model load failed: {e}")
        return None

@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarizer():
    """
    Loads the summarization pipeline and caches it.
    """
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Summarization model load failed: {e}")
        return None

def transcribe_audio(file_path, model):
    """Transcribes the audio file using the Whisper model."""
    try:
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return f"[ERROR: Transcription failed] {e}"

def summarize_text(text, summarizer, max_length=130, min_length=30, bullet_style=True):
    """Summarizes the given text."""
    try:
        # Check if the text is long enough to summarize
        if len(text.split()) < min_length:
            return "The text is too short to summarize. Please provide more content."

        summary_list = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,  # Set to False for reproducible results
            num_return_sequences=1
        )
        summary = summary_list[0]["summary_text"]
        if bullet_style:
            sentences = summary.split(". ")
            summary = "\n".join([f"• {s.strip()}" for s in sentences if s])
        return summary
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        return f"[ERROR: Summarization failed] {e}"

def send_email(recipient, subject, body, sender_email, sender_password):
    """Sends an email using the provided SMTP details."""
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return str(e)

# -------------------------
# Streamlit UI
# -------------------------
st.title("📝 AI Notes Summarizer & Sharer")
st.write("Upload an audio file or paste text to get a summary.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "flac", "aac", "ogg"])
text_input = st.text_area("Or Paste Text to Summarize", height=150)

st.markdown("---")
st.subheader("Summary Settings")
max_len = st.slider("Max Summary Length", 50, 500, 130)
min_len = st.slider("Min Summary Length", 10, 200, 30)
bullet_mode = st.checkbox("Format Summary in Bullet Points", value=True)

# -------------------------
# Main Logic
# -------------------------
# Load models at the beginning of the script
whisper_model = load_whisper_model()
summarizer = load_summarizer()

# Session state to store transcription and summary
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Handle audio file upload and transcription
if uploaded_file:
    # Use tempfile to handle temporary audio file creation safely
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as temp_audio:
            temp_audio.write(uploaded_file.getbuffer())
            temp_audio_path = temp_audio.name

        st.audio(temp_audio_path)
        if st.button("Transcribe & Summarize Audio"):
            if whisper_model:
                with st.spinner("Transcribing... this might take a while."):
                    st.session_state.transcription = transcribe_audio(temp_audio_path, whisper_model)
                with st.spinner("Summarizing..."):
                    st.session_state.summary = summarize_text(
                        st.session_state.transcription, summarizer, max_len, min_len, bullet_mode
                    )
            else:
                st.error("Whisper model is not available. Check the logs for loading errors.")
    finally:
        # Ensure the temporary file is deleted after use
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Handle text input and summarization
if text_input.strip() and st.button("Summarize Pasted Text"):
    st.session_state.transcription = text_input
    with st.spinner("Summarizing..."):
        st.session_state.summary = summarize_text(
            st.session_state.transcription, summarizer, max_len, min_len, bullet_mode
        )

# -------------------------
# Display results
# -------------------------
st.markdown("---")
if st.session_state.transcription:
    st.subheader("Transcription")
    st.write(st.session_state.transcription)
    st.download_button("Download Transcription", data=st.session_state.transcription, file_name="transcription.txt")

if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)
    st.download_button("Download Summary", data=st.session_state.summary, file_name="summary.txt")

# -------------------------
# Send summary via email
# -------------------------
if st.session_state.summary:
    st.markdown("---")
    st.subheader("Send Summary via Email")
    recipient = st.text_input("Recipient Email", key="recipient_input")
    subject = st.text_input("Email Subject", "Your AI-Generated Summary", key="subject_input")
    
    sender_email, sender_password = load_mail_credentials()

    if st.button("Send Email"):
        if not recipient:
            st.warning("Please enter a recipient email address.")
        elif sender_email and sender_password:
            with st.spinner("Sending email..."):
                result = send_email(recipient, subject, st.session_state.summary, sender_email, sender_password)
            if result is True:
                st.success("Email sent successfully!")
            else:
                st.error(f"Failed to send email. Check your credentials and network connection.")
        else:
            st.error("Email sending is not configured. Please add credentials to Streamlit secrets.")
