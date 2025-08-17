import streamlit as st
import whisper
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from transformers import pipeline
import random

# -------------------------
# Load email credentials
# -------------------------
def load_mail_credentials():
    try:
        sender_email = st.secrets["mail"]["MAIL_SENDER_EMAIL"]
        sender_password = st.secrets["mail"]["MAIL_SENDER_PASS"]
        return sender_email, sender_password
    except Exception as e:
        st.error(f"Error loading mail credentials: {e}")
        return None, None

# -------------------------
# Helper Functions
# -------------------------
def transcribe_audio(file_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"[ERROR: Transcription failed] {e}"

# Hugging Face summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30, bullet_style=True):
    try:
        # Introduce slight randomness to avoid duplicate summaries
        top_k = random.randint(40, 60)
        top_p = round(random.uniform(0.9, 0.95), 2)

        summary_list = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1
        )
        summary = summary_list[0]["summary_text"]

        if bullet_style:
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

# -------------------------
# Session state initialization
# -------------------------
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False

# -------------------------
# Streamlit UI
# -------------------------
st.title("📝 AI Notes Summarizer & Sharer")
st.write("Upload audio or paste text → transcribe/summarize → regenerate if needed → share via email.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "flac", "aac", "ogg"])
text_input = st.text_area("Or Paste Text to Summarize", height=150)

max_len = st.slider("Max Summary Length", min_value=50, max_value=500, value=130)
min_len = st.slider("Min Summary Length", min_value=10, max_value=200, value=30)
bullet_mode = st.checkbox("Format Summary in Bullet Points", value=True)

# -------------------------
# Show Transcribe button only after input
# -------------------------
if uploaded_file or text_input.strip() != "":
    if uploaded_file:
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_file_path)

    if st.button("Transcribe & Summarize"):
        # Transcription
        if uploaded_file:
            st.session_state.transcription = transcribe_audio(temp_file_path)
        else:
            st.session_state.transcription = text_input

        # Summary
        st.session_state.summary = summarize_text(
            st.session_state.transcription, max_length=max_len, min_length=min_len, bullet_style=bullet_mode
        )
        st.session_state.summary_generated = True

# -------------------------
# Display transcription
# -------------------------
if st.session_state.transcription:
    st.subheader("Transcription")
    st.write(st.session_state.transcription)
    st.download_button("Download Transcription", data=st.session_state.transcription, file_name="transcription.txt")

# -------------------------
# Display summary and regenerate
# -------------------------
if st.session_state.summary_generated:
    st.subheader("Summary")
    st.write(st.session_state.summary)
    st.download_button("Download Summary", data=st.session_state.summary, file_name="summary.txt")

    if st.button("Regenerate Summary"):
        st.session_state.summary = summarize_text(
            st.session_state.transcription, max_length=max_len, min_length=min_len, bullet_style=bullet_mode
        )
        st.success("Summary regenerated!")

# -------------------------
# Send email section
# -------------------------
if st.session_state.summary_generated:
    st.subheader("Send Summary via Email")
    recipient = st.text_input("Recipient Email", key="recipient_input")
    subject = st.text_input("Email Subject", "Your Summary", key="subject_input")

    sender_email, sender_password = load_mail_credentials()

    if sender_email is None or sender_password is None:
        st.error("Cannot send email: Sender credentials not loaded from secrets")
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
