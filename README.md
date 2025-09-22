# 🎤 Voice FAQ Assistant

👉 **[Live Demo(not working yet)](https://your-live-site-link.com)** 👈

AI-powered voice assistant that answers FAQs using speech recognition, natural language processing, and text-to-speech.

## ✨ Features

* 🎙️ Voice recognition (record questions)
* 🤖 Smart FAQ matching with embeddings
* 🔊 Text-to-speech responses
* 📊 Real-time audio visualization
* 📱 Responsive web interface

## 🛠️ Tech Stack

* **Backend**: FastAPI
* **AI/ML**: Whisper, Sentence Transformers, spaCy
* **Audio**: Librosa, PyDub, Noise Reduction
* **TTS**: Google gTTS
* **Frontend**: JavaScript + HTML5

## 🚀 Getting Started

1. **Clone repo**

   ```bash
   git clone https://github.com/giftkalu/Voice-Assistant-for-FAQ.git
   cd Voice-Assistant-for-FAQ
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Run locally**

   ```bash
   python main.py
   ```

   Open [http://localhost:8000](http://localhost:8000).

## 📁 Project Structure

```
voice-faq-assistant/
├── main.py            # FastAPI backend
├── fva.html           # Web interface
├── sample_faq.json    # Example FAQs
├── requirements.txt   # Dependencies
└── README.md
```

## 📝 FAQ Data Format

CSV file with columns:

```csv
question,answer
"What is Cowrywise?","Cowrywise is a digital savings and investment platform..."
```

## 🔒 Security

* No audio stored (processed & deleted immediately)
* File validation (only audio files)
* CORS protection enabled

## ⚠️ Known Issues

* Works best with WAV/MP3/M4A
* Requires microphone permission
* Latency may occur on first load

## 👨‍💻 Author

**Gift Kalu** – *Your Creative Data Scientist*

* LinkedIn: [Gift Kalu](https://www.linkedin.com/in/gift-kalu)

---

⭐ Star this repo if you like it!

---
