import os
import torch
import random
import pandas as pd
import librosa
from itertools import combinations
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_metric
import numpy as np
import gradio as gr
import torch.nn.functional as F

# Load WER metric
wer_metric = load_metric("wer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU


Final_model_path = '/notebooks/FTbase_model/epoch_5'
# Load the model and processor
model = Wav2Vec2ForCTC.from_pretrained(Final_model_path).to('cuda')
processor = Wav2Vec2Processor.from_pretrained(Final_model_path)

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


callback = gr.CSVLogger()

# Function to transcribe audio
def transcribe(audio_input, model=model, processor=processor):
    if audio_input is None:
        return "Error: No audio file received. Please record and submit audio."

    try:
        # Check if input is a file path (upload) or a NumPy array (microphone)
        if isinstance(audio_input, str):
            y, sr = librosa.load(audio_input, sr=16_000)
        else:  # If audio from microphone
            y, sr = audio_input  # NumPy array and sample rate
        
        # Process the audio file with the processor and move inputs to the correct device
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        # Assuming `inputs` is your input tensor, move it to the same device
        inputs = inputs.to(device)

        # Pass the inputs through the model
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        # Get the predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the ids to get the predicted sentence
        predicted_sentence = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return predicted_sentence[0]  # Return the predicted sentence
    except Exception as e:
        print(f"Error processing audio: {e}")
        return "Error: Unable to process the audio file. Please try again."


# Function to handle corrections
def alert_flag_submission():
    gr.Info('Flag submitted')
import gradio as gr
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import os

callback = gr.CSVLogger()

# Function to transcribe audio
def transcribe(audio_input, model=model, processor=processor):
    if audio_input is None:
        return "Error: No audio file received. Please record and submit audio."

    try:
        # Check if input is a file path (upload) or a NumPy array (microphone)
        if isinstance(audio_input, str):
            y, sr = librosa.load(audio_input, sr=16_000)
        else:  # If audio from microphone
            y, sr = audio_input  # NumPy array and sample rate
        
        # Process the audio file with the processor and move inputs to the correct device
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        # Assuming `inputs` is your input tensor, move it to the same device
        inputs = inputs.to(device)

        # Pass the inputs through the model
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        # Get the predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the ids to get the predicted sentence
        predicted_sentence = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return predicted_sentence[0]  # Return the predicted sentence
    except Exception as e:
        print(f"Error processing audio: {e}")
        return "Error: Unable to process the audio file. Please try again."


# Function to handle corrections
def alert_flag_submission():
    gr.Info('Flag submitted')

# Create the Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Audio Transcription and Correction")
    
    # State to store the username
    username_state = gr.State(value=None)  # Initialize state to None
    
    # Username input
    username_input = gr.Textbox(label="Username", placeholder="Enter your username...", interactive=True)
    
    # Audio input for recording or uploading
    audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Record Your Audio")
    
    # Output for the transcription
    transcription_output = gr.Textbox(label="Transcription", interactive=False)
    
    # Correction input box
    correction_input = gr.Textbox(label="Corrected Transcription", placeholder="Edit the transcription here...")
    
    # Button to submit the correction
    submit_button = gr.Button("Submit Correction")
    
    # This needs to be called at some point prior to the first call to callback.flag()
    callback.setup([audio_input, transcription_output, correction_input, username_input], "flagged_vocal_transcripts")
    
    # Define the interaction
    audio_input.change(fn=transcribe, inputs=audio_input, outputs=transcription_output)
        
    # Apply Flagging on the needed components
    submit_button.click(lambda *args: callback.flag(list(args)), [audio_input, transcription_output, correction_input, username_input], None, preprocess=False)
    # Alert user that message has been submitted
    submit_button.click(fn=alert_flag_submission, inputs=None, outputs=None)

    # Clear the correction input after submitting new audio
    audio_input.change(fn=lambda: "", inputs=None, outputs=correction_input)
    
# Launch the Gradio interface
demo.launch(share=True)