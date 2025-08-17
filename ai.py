import streamlit as st
import whisper
from transformers import pipeline
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------------- CONFIG -------------------------
# Load email credentials
def load_mail_credentials():
    try:
        sender_email = st.secrets["mail"]["MAIL_SENDER_EMAIL"]
        sender_password = st.secrets["mail"]["MAIL_SENDER_PASS"]
        return sender_email, sender_password
    except Exception as e:
        st.error(f"❌ Error loading email credentials: {e}")
        return None, None

# ------------------------- MODEL LOAD -------------------------
@st.cache_resource
def load_models():
    transcriber = whisper.load_model("base")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return transcriber, summarizer

transcriber_model, summarizer_model = load_models()

# ------------------------- FUNCTIONS -------------------------
def transcribe_audio(file_path):
    try:
        result = transcriber_model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"[❌ Transcription failed] {e}"

def summarize_text(text, max_length=130, min_length=30, bullet_style=True):
    try:
        summary_output = summarizer_model(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        summary = summary_output[0]["summary_text"]
        if bullet_style:
            sentences = summary.split(". ")
            summary = "\n".join([f"• {s.strip()}" for s in sentences if s])
        return summary
    except Exception as e:
        return f"[❌ Summarization failed] {e}"

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

# ------------------------- STREAMLIT UI -------------------------
st.title("📝 AI Notes Summarizer & Sharer")
st.caption("Upload audio or paste text → transcribe/summarize → regenerate if needed → share via email.")

uploaded_file = st.file_uploader("🎵 Upload Audio", type=["wav", "mp3", "m4a", "flac", "aac", "ogg"])
text_input = st.text_area("📋 Or Paste Text to Summarize", height=150)

# Show config only when input is present
if uploaded_file or text_input.strip():
    max_len = st.slider("📏 Max Summary Length", 50, 500, 130)
    min_len = st.slider("📏 Min Summary Length", 10, 200, 30)
    bullet_mode = st.checkbox("• Format Summary in Bullet Points", value=True)
else:
    st.info("👉 Upload audio or paste text to activate settings and buttons.")
    max_len = 130
    min_len = 30
    bullet_mode = True

# State Setup
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "show_summary_button" not in st.session_state:
    st.session_state.show_summary_button = False

# ------------------ TRANSCRIBE ------------------
temp_path = None
if uploaded_file:
    temp_path = f"./temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(temp_path)

    if st.button("🧠 Transcribe Audio"):
        with st.spinner("Transcribing..."):
            st.session_state.transcription = transcribe_audio(temp_path)
            st.session_state.show_summary_button = True

# ------------------ TEXT INPUT ------------------
if text_input.strip() != "":
    st.session_state.transcription = text_input
    st.session_state.show_summary_button = True

# ------------------ SHOW TRANSCRIPTION ------------------
if st.session_state.transcription:
    st.subheader("🗣️ Transcription")
    st.write(st.session_state.transcription)
    st.download_button("⬇️ Download Transcription", data=st.session_state.transcription, file_name="transcription.txt")

# ------------------ GENERATE SUMMARY ------------------
if st.session_state.show_summary_button:
    if st.button("🧾 Generate Summary"):
        with st.spinner("Summarizing..."):
            st.session_state.summary = summarize_text(
                st.session_state.transcription,
                max_length=max_len,
                min_length=min_len,
                bullet_style=bullet_mode
            )

# ------------------ DISPLAY SUMMARY ------------------
if st.session_state.summary:
    st.subheader("🧾 Summary")
    st.write(st.session_state.summary)
    st.download_button("⬇️ Download Summary", data=st.session_state.summary, file_name="summary.txt")

    if st.button("🔄 Regenerate Summary"):
        with st.spinner("Re-summarizing..."):
            st.session_state.summary = summarize_text(
                st.session_state.transcription,
                max_length=max_len,
                min_length=min_len,
                bullet_style=bullet_mode
            )
            st.success("✅ Summary regenerated!")

# ------------------ EMAIL SECTION ------------------
if st.session_state.summary:
    st.subheader("📧 Send Summary via Email")
    recipient = st.text_input("To (Recipient Email)", key="recipient_input")
    subject = st.text_input("Subject", "Your AI Summary", key="subject_input")

    sender_email, sender_password = load_mail_credentials()

    if sender_email and sender_password:
        if st.button("📤 Send Email"):
            with st.spinner("Sending email..."):
                result = send_email(recipient, subject, st.session_state.summary, sender_email, sender_password)
                if result == True:
                    st.success("✅ Email sent successfully!")
                else:
                    st.error(f"❌ Failed to send email: {result}")

# ------------------ CLEANUP ------------------
try:
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
except:
    pass
