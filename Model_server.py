from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import librosa
from transformers import pipeline

# Load your custom ASR model
asr = pipeline("automatic-speech-recognition", model="/notebooks/FTbase_model_e5")

# Initialize FastAPI app
app = FastAPI()

# Endpoint for transcribing audio
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Read audio file
        audio_data = await file.read()
        # Decode audio using Librosa
        y, sr = librosa.load(file.filename, sr=None)

        # Resample audio to 16kHz if necessary
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Perform transcription
        transcription = asr({"sampling_rate": sr, "raw": y})["text"]
        transcription = transcription.replace("<unk>", "").strip()

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
