# Gen-AI Project: Automated AI Podcast Pipeline

## Overview
This project is an end-to-end automated pipeline for generating AI-driven multimedia content. It creates a professional, animated podcast featuring two characters (Host and Expert) discussing any given topic. The system integrates Large Language Models (LLMs), Text-to-Speech (TTS), Diffusion Models, and AI-driven Animation.

## 🚀 Pipeline Workflow
The pipeline operates as a linear dependency chain, orchestrated by `main.py`. Each step depends on the output of the previous one:

1.  **Scripting:** LLM generates a dialogue script based on the input topic.
2.  **Visuals:** Diffusion model generates high-quality avatar portraits.
3.  **Voice:** TTS engine converts dialogue text into individual audio clips.
4.  **Animation:** Lip-sync model animates the avatars using the generated audio.
5.  **Assembly:** Video editor stitches clips into a professional side-by-side layout.

---

## 📂 Source Code Analysis (`src/`)

### 1. `generator_script.py`
*   **Role:** The "Brain" of the pipeline.
*   **Model:** Ollama (Llama 3).
*   **Input:** A string representing the podcast topic.
*   **Output:** A structured JSON list of dialogue objects (e.g., `[{"speaker": "Host", "text": "..."}]`).
*   **Logic:** Includes advanced regex cleaning to ensure the LLM output is valid JSON.

### 2. `generator_avatar.py`
*   **Role:** Character Creation.
*   **Model:** SDXL-Turbo (Optimized for T4 GPUs).
*   **Input:** Hardcoded descriptive prompts for "Host" and "Expert".
*   **Output:** Two 512x512 PNG images saved to `data/avatars/`.
*   **Optimization:** Aggressively manages VRAM to allow high-resolution generation on limited hardware.

### 3. `generator_audio.py`
*   **Role:** Vocal Synthesis.
*   **Model:** Kokoro-ONNX (v1.0).
*   **Input:** The JSON script from Step 1.
*   **Output:** A series of `.wav` files in `data/audio/`, named numerically (e.g., `001_Host.wav`).
*   **Features:** Assigns distinct male/female voices to characters.

### 4. `generator_animation.py`
*   **Role:** Lip-Sync & Motion.
*   **Model:** SadTalker (with GFPGAN face enhancement).
*   **Input:** The `.wav` audio files and the `.png` avatar images.
*   **Output:** Individual `.mp4` video clips in `data/animations/`.
*   **Speed:** Uses a custom batch script to load models once and process all 60+ dialogue lines efficiently.

### 5. `video_montage.py` & `advanced_montage.py`
*   **Role:** Final Production.
*   **Tool:** MoviePy with FFmpeg (NVENC support).
*   **Input:** The `.mp4` animation clips from Step 4.
*   **Output:** A final synchronized video in `outputs/`.
*   **Advanced Features:** `advanced_montage.py` adds dynamic highlighting (dimming the non-speaking character), hardware-accelerated encoding, and professional text overlays.

---

## 📊 Data Flow & Storage
- **`data/avatars/`**: Stores the generated character images.
- **`data/audio/`**: Stores the TTS-generated speech clips.
- **`data/animations/`**: Stores the individual animated dialogue segments.
- **`outputs/`**: Contains the final high-definition MP4 podcast.
- **`models/`**: (Ignored by Git) Contains the heavy AI model weights (Kokoro, SadTalker, etc.).

---

## 🛠️ Technical Stack
- **Language:** Python 3.10+
- **AI Frameworks:** PyTorch, Hugging Face Transformers, Diffusers, Accelerate.
- **Video/Audio:** MoviePy, FFmpeg, Kokoro-ONNX, SoundFile.
- **LLM Engine:** Ollama.

## ⚙️ Setup & Usage

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download model weights:
   ```bash
   python download_sadtalker_weights.py
   ```
3. Set up Hugging Face token:
   ```bash
   export HF_TOKEN="your_token"
   ```

### Running the Pipeline
Run the full pipeline with a custom topic:
```bash
python main.py --topic "The Future of Space Exploration" --advanced
```

To re-run only the final montage (if data already exists):
```bash
python src/run_advanced_montage.py
```

## Author
**Yahya Bouchak**
*Student in Master SIIA (Systèmes Intelligents et Intelligence Artificielle)*
