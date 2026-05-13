# Gen-AI Project

## Overview
This project is an automated pipeline for generating AI-driven multimedia content, including podcast scripts, audio, avatars, and animations. It integrates various AI models to create a seamless content generation workflow.

## Features
- **Script Generation:** Automatically generates podcast-style scripts.
- **Audio Synthesis:** Converts generated scripts into high-quality audio files.
- **Avatar Generation:** Creates visual avatars for content representation.
- **Animation:** Animates avatars to match the generated audio.
- **Video Montage:** Combines all elements into a final video output.

## Technical Stack
- **Language:** Python
- **Frameworks:** PyTorch, Hugging Face Transformers, Accelerate
- **Core Components:**
  - `src/generator_script.py`
  - `src/generator_audio.py`
  - `src/generator_avatar.py`
  - `src/generator_animation.py`
  - `src/video_montage.py`

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your Hugging Face token as an environment variable:
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```
3. Run the main pipeline:
   ```bash
   python main.py
   ```

## Author
**Yahya Boucha**
*Student in Master SIIA (Systèmes Intelligents et Intelligence Artificielle)*

---
*Created for the Gen-AI project repository.*
