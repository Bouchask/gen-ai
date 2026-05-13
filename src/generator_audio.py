import os
import logging
import soundfile as sf
from kokoro_onnx import Kokoro
import torch
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_kokoro_models(model_dir="models/kokoro"):
    """
    Downloads the Kokoro ONNX model and voices if not present.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "kokoro-v1.0.onnx")
    voices_path = os.path.join(model_dir, "voices-v1.0.bin")
    
    # URLs for Kokoro v1.0
    model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    
    for url, path in [(model_url, model_path), (voices_url, voices_path)]:
        if not os.path.exists(path):
            logging.info(f"Downloading {os.path.basename(path)}...")
            r = requests.get(url, allow_redirects=True)
            with open(path, 'wb') as f:
                f.write(r.content)
    
    return model_path, voices_path

def generate_audio_files(script, output_dir="data/audio"):
    """
    Generates .wav files for each line in the script using Kokoro TTS.
    """
    model_path, voices_path = download_kokoro_models()
    
    logging.info("Initializing Kokoro TTS...")
    kokoro = Kokoro(model_path, voices_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    audio_paths = []
    
    # Defined speaker voices (Kokoro has 'af_bella', 'am_adam', etc.)
    voices = {
        "Host": "am_adam",   # Male American
        "Expert": "af_bella" # Female American
    }
    
    for i, line in enumerate(script):
        speaker = line['speaker']
        text = line['text']
        filename = f"{i:03d}_{speaker}.wav"
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            logging.info(f"⏭️ Audio for line {i} already exists. Skipping.")
            audio_paths.append({"speaker": speaker, "audio_path": output_path, "text": text})
            continue
            
        logging.info(f"Generating audio for {speaker}: {text[:50]}...")
        
        try:
            samples, sample_rate = kokoro.create(
                text, 
                voice=voices.get(speaker, "af_bella"), 
                speed=1.0, 
                lang="en-us"
            )
            sf.write(output_path, samples, sample_rate)
            audio_paths.append({"speaker": speaker, "audio_path": output_path, "text": text})
        except Exception as e:
            logging.error(f"Failed to generate audio for line {i}: {e}")
            
    return audio_paths

if __name__ == "__main__":
    sample_script = [
        {"speaker": "Host", "text": "Hello and welcome to the AI Revolution podcast!"},
        {"speaker": "Expert", "text": "It is great to be here to discuss industrial automation."}
    ]
    generate_audio_files(sample_script)
