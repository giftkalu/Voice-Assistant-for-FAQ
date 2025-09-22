import os
import io
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Your existing imports and functions from the notebook
import whisper
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your models (copy from notebook)
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your FAQ data - CHANGE THIS PATH
faq_df = pd.read_csv("500faq.csv")  # Upload this to cloud storage
nlp = spacy.load("en_core_web_sm")

# Copy your existing functions from notebook
def preprocess_question(q):
    doc = nlp(q.lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(tokens)

def find_best_faq_match(user_question, threshold=0.7):
    def get_best_match(query):
        user_embedding = embedding_model.encode([query])
        similarities = cosine_similarity(user_embedding, question_embeddings)[0]

        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        return best_idx, best_similarity

    # --- Strategy 1: Original question
    best_idx, best_similarity = get_best_match(user_question)
    if best_similarity >= threshold:
        return {
            'question': faq_df.iloc[best_idx]['question'],
            'answer': faq_df.iloc[best_idx]['answer'],
            'similarity': float(best_similarity),
            'index': best_idx
        }

    # --- Strategy 2: Append "cowrywise"
    best_idx, best_similarity = get_best_match(user_question + " cowrywise")
    if best_similarity >= threshold:
        return {
            'question': faq_df.iloc[best_idx]['question'],
            'answer': faq_df.iloc[best_idx]['answer'],
            'similarity': float(best_similarity),
            'index': best_idx
        }

    # --- Strategy 3: Replace "you" or "u" with "cowrywise"
    replaced_question = (
        user_question.replace(" you ", " cowrywise ")
        .replace(" u ", " cowrywise ")
        .replace("you", "cowrywise")  # catch edge cases
        .replace("u", "cowrywise")
    )
    if replaced_question != user_question:  # only retry if something actually changed
        best_idx, best_similarity = get_best_match(replaced_question)
        if best_similarity >= threshold:
            return {
                'question': faq_df.iloc[best_idx]['question'],
                'answer': faq_df.iloc[best_idx]['answer'],
                'similarity': float(best_similarity),
                'index': best_idx
            }
    if best_similarity >= 0.4:
        return {
            'question': faq_df.iloc[best_idx]['question'],
            'answer': "This seems related to your question: " + faq_df.iloc[best_idx]['answer'] + "Hope this helps",
            'similarity': float(best_similarity),
            'index': best_idx
        }

    # --- Fallback: return best guess but with unsure answer
    return {
        'question': faq_df.iloc[best_idx]['question'],
        'similarity': float(best_similarity),
        'answer': "I'm not sure how to answer that right now. Try rephrasing your question."
    }

def preprocess_and_transcribe(audio_path):
    try:
        # Step 1: Load audio with fallback options
        y, sr = librosa.load(audio_path, sr=None)
        
        # Step 2: Reduce noise
        try:
            reduced = nr.reduce_noise(y=y, sr=sr)
            sf.write("cleaned.wav", reduced, sr)
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            # Skip noise reduction, use original
            sf.write("cleaned.wav", y, sr)
        
        # Step 3: Normalize
        try:
            sound = AudioSegment.from_wav("cleaned.wav")
            normalized = effects.normalize(sound)
            normalized.export("normalized.wav", format="wav")
            final_path = "normalized.wav"
        except Exception as e:
            print(f"Normalization failed: {e}")
            # Use cleaned audio instead
            final_path = "cleaned.wav"
        
        # Step 4: Transcribe with bias
        result = whisper_model.transcribe(
            final_path, 
            language="en", 
            initial_prompt="balance, account, payment, transaction, "
                          "mutual funds, savings, opay, direct, interest, "
                          "customer, earnings, Cowrywise, fintech, plan, "
                          "rate, returns, bank, naira"
        )
        
        transcript = result["text"]
        
        # Clean up temp files
        for temp_file in ["cleaned.wav", "normalized.wav"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        return transcript
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Could not process audio. Please try again."
# Initialize
faq_df['clean_question'] = faq_df['question'].apply(preprocess_question)
questions = faq_df['clean_question'].tolist()
question_embeddings = embedding_model.encode(questions)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("fva.html")

@app.post("/voice-assist")
async def voice_assist(audio: UploadFile = File(...)):
    try:
        content = await audio.read()
        
        # Save audio file
        with open("audio.wav", "wb") as f:
            f.write(content)
        
        # Transcribe
        transcription = preprocess_and_transcribe("audio.wav")
        print(f"Transcribed: {transcription}")
        
        # Get answer
        processed_question = preprocess_question(transcription)
        result = find_best_faq_match(processed_question)
        
        # Convert to speech
        tts = gTTS(result['answer'])
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Clean up
        os.remove("audio.wav")
        
        return StreamingResponse(mp3_fp, media_type="audio/mpeg")
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))  # use Render's PORT if available
    uvicorn.run(app, host="0.0.0.0", port=port)
