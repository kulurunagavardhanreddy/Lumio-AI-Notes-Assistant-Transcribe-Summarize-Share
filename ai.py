import streamlit as st
import whisper
from transformers import pipeline
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# -------------------------
# Email credentials from Streamlit secrets
# -------------------------
def load_mail_credentials():
    try:
        sender_email = st.secrets["mail"]["MAIL_SENDER_EMAIL"]
        sender_password = st.secrets["mail"]["MAIL_SENDER_PASS"]
        return sender_email, sender_password
    except Exception as e:
        st.error(f"Error loading mail credentials from secrets: {e}")
        return None, None

# -------------------------
# Helper functions
# -------------------------
@st.cache_resource(show_spinner=True)
def load_whisper_model(model_name="base"):
    try:
        return whisper.load_model(model_name)
    except Exception as e:
        st.error(f"Whisper model load failed: {e}")
        return None

@st.cache_resource(show_spinner=True)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def transcribe_audio(file_path, model):
    try:
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"[ERROR: Transcription failed] {e}"

def summarize_text(text, summarizer, max_length=130, min_length=30, bullet_style=True):
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
# Streamlit UI
# -------------------------
st.title("📝 AI Notes Summarizer & Sharer")
st.write("Upload audio or paste text → transcribe/summarize → share via email.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "flac", "aac", "ogg"])
text_input = st.text_area("Or Paste Text to Summarize", height=150)

max_len = st.slider("Max Summary Length", 50, 500, 130)
min_len = st.slider("Min Summary Length", 10, 200, 30)
bullet_mode = st.checkbox("Format Summary in Bullet Points", value=True)

# -------------------------
# Load models
# -------------------------
whisper_model = load_whisper_model("base")
summarizer = load_summarizer()

# -------------------------
# Session state
# -------------------------
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# -------------------------
# Audio transcription
# -------------------------
if uploaded_file is not None:
    temp_file_path = f"./temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(temp_file_path)

    if st.button("Transcribe & Summarize Audio"):
        if whisper_model:
            st.session_state.transcription = transcribe_audio(temp_file_path, whisper_model)
            st.session_state.summary = summarize_text(
                st.session_state.transcription, summarizer, max_len, min_len, bullet_mode
            )
        else:
            st.error("Whisper model not loaded!")

# -------------------------
# Summarize pasted text
# -------------------------
if text_input.strip() and st.button("Summarize Pasted Text"):
    st.session_state.transcription = text_input
    st.session_state.summary = summarize_text(
        st.session_state.transcription, summarizer, max_len, min_len, bullet_mode
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
# Send summary via email
# -------------------------
if st.session_state.summary:
    st.subheader("Send Summary via Email")
    recipient = st.text_input("Recipient Email", key="recipient_input")
    subject = st.text_input("Email Subject", "Your Summary", key="subject_input")
    sender_email, sender_password = load_mail_credentials()

    if sender_email and sender_password:
        if st.button("Send Email"):
            result = send_email(recipient, subject, st.session_state.summary, sender_email, sender_password)
            if result is True:
                st.success("Email sent successfully!")
            else:
                st.error(f"Failed to send email: {result}")
    else:
        st.error("Sender credentials not loaded!")

# -------------------------
# Cleanup temp files
# -------------------------
try:
    if uploaded_file and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
except:
    pass
