import streamlit as st
import whisper
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import toml
import yaml
from transformers import pipeline

# -------------------------
# Load email credentials from config.toml
# -------------------------
CONFIG_FILE = ".streamlit/config.toml"

def load_mail_credentials():
    try:
        # Streamlit secrets automatically reads from .streamlit/secrets.toml or deployed secrets
        sender_email = st.secrets["mail"]["MAIL_SENDER_EMAIL"]
        sender_password = st.secrets["mail"]["MAIL_SENDER_PASS"]
        return sender_email, sender_password
    except Exception as e:
        st.error(f"Error loading mail credentials from secrets: {e}")
        return None, None

# -------------------------
# Helper Functions
# -------------------------
def transcribe_audio(file_path):
    try:
        model = whisper.load("base")  # tiny/small/medium/large
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"[ERROR: Transcription failed] {e}"

# Hugging Face summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30, bullet_style=True):
    """
    Summarize text using Hugging Face pipeline.
    - max_length & min_length control summary length
    - bullet_style formats the summary in bullet points
    """
    try:
        summary_list = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        summary = summary_list[0]["summary_text"]

        if bullet_style:
            # Split sentences and add bullets
            sentences = summary.split(". ")
            summary = "\n".join([f"• {s.strip()}" for s in sentences if s])
        return summary
    except Exception as e:
        return f"[ERROR: Summarization failed] {e}"

def send_email(recipient, subject, body, sender_email, sender_password):
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
        return str(e)

def save_text_to_file(text, filename):
    with open(filename, "w") as f:
        f.write(text)
    return filename

# -------------------------
# Initialize session state
# -------------------------
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# -------------------------
# Streamlit UI
# -------------------------
st.title("📝 AI Notes Summarizer & Sharer")
st.write("Upload audio or paste text → transcribe/summarize → share via email.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "flac", "aac", "ogg"])
text_input = st.text_area("Or Paste Text to Summarize", height=150)

max_len = st.slider("Max Summary Length", min_value=50, max_value=500, value=130)
min_len = st.slider("Min Summary Length", min_value=10, max_value=200, value=30)
bullet_mode = st.checkbox("Format Summary in Bullet Points", value=True)

# -------------------------
# Audio Transcription
# -------------------------
if uploaded_file is not None:
    temp_file_path = f"./temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_file_path)

    if st.button("Transcribe & Summarize Audio"):
        st.session_state.transcription = transcribe_audio(temp_file_path)
        st.session_state.summary = summarize_text(
            st.session_state.transcription, max_length=max_len, min_length=min_len, bullet_style=bullet_mode
        )

# -------------------------
# Summarize Pasted Text
# -------------------------
if text_input.strip() != "" and st.button("Summarize Pasted Text"):
    st.session_state.transcription = text_input
    st.session_state.summary = summarize_text(
        st.session_state.transcription, max_length=max_len, min_length=min_len, bullet_style=bullet_mode
    )

# -------------------------
# Show transcription and summary
# -------------------------
if st.session_state.transcription:
    st.subheader("Transcription")
    st.write(st.session_state.transcription)
    st.download_button("Download Transcription", data=st.session_state.transcription, file_name="transcription.txt")

if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)
    st.download_button("Download Summary", data=st.session_state.summary, file_name="summary.txt")

# -------------------------
# Send Summary via Email
# -------------------------
if st.session_state.summary:
    st.subheader("Send Summary via Email")
    recipient = st.text_input("Recipient Email", key="recipient_input")
    subject = st.text_input("Email Subject", "Your Summary", key="subject_input")

    sender_email, sender_password = load_mail_credentials()

    if sender_email is None or sender_password is None:
        st.error("Cannot send email: Sender credentials not loaded from config.toml")
    elif st.button("Send Email"):
        result = send_email(recipient, subject, st.session_state.summary, sender_email, sender_password)
        if result == True:
            st.success("Email sent successfully!")
        else:
            st.error(f"Failed to send email: {result}")

# -------------------------
# Cleanup temp files
# -------------------------
try:
    if uploaded_file and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
except:
    pass
