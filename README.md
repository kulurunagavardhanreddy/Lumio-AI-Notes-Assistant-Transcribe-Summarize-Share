# 🎧 Lumio AI Notes Assistant: Transcribe, Summarize & Share

A simple and fast **Audio & Text Summarizer** using Hugging Face Transformers and Streamlit.  
Upload audio files or paste text, generate a summary in bullet points, and optionally email it.

[View Demo on Streamlit](https://lumio-ai-notes-assistant-transcribe-summarize-share-xbs2nyk5lg.streamlit.app/)

---

## 📦 Features

- **Audio Transcription**: Upload `.mp3`, `.wav`, `.m4a`, or `.ogg` files.  
- **Text Summarization**: Automatically summarize transcribed text in a readable, bullet-point style.  
- **Direct Text Input**: Paste text to summarize directly.  
- **Email Summary**: Send the generated summary via email.  
- **Clean UI**: Streamlit interface with buttons for transcription and summarization.

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/) – Interactive web app framework.  
- [Transformers](https://huggingface.co/transformers/) – Hugging Face pipeline for ASR and summarization.  
- [Python](https://www.python.org/) – Backend language.  
- [SMTP] (Gmail) – Send emails with summaries.  

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/kulurunagavardhanreddy/Lumio-AI-Notes-Assistant-Transcribe-Summarize-Share.git
cd Lumio-AI-Notes-Assistant-Transcribe-Summarize-Share
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Set email credentials (optional for sending emails)

MAIL_SENDER_EMAIL = "your-email@gmail.com"
MAIL_SENDER_PASS = "your-app-password"

* **Note: For Gmail, you might need to generate an App Password.

### 4. Run the app

```
streamlit run app.py
```

#### Usage

* **1. Upload an audio file or paste your text.

* **2. Click Transcribe Audio (for audio files).

* **3. Click Generate Summary to get a readable summary in bullet points.

* **4. Optionally, enter a recipient email and click Send Email.

### Links

````
GitHub Repository: https://github.com/kulurunagavardhanreddy/Lumio-AI-Notes-Assistant-Transcribe-Summarize-Share
```

### Hugging Face Models:

```
facebook/wav2vec2-base-960h
```

```
facebook/bart-large-cnn
```

```
Streamlit: https://streamlit.io/
```