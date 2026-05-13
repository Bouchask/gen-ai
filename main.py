import os
# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import argparse
import os
import shutil
import torch
import gc
from huggingface_hub import login
from accelerate import Accelerator
from src.generator_script import generate_podcast_script
from src.generator_audio import generate_audio_files
from src.generator_avatar import generate_avatars
from src.generator_animation import animate_avatars
from src.video_montage import create_montage

# Initialize Accelerator
accelerator = Accelerator()
device = accelerator.device

# Authenticate with Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    logging.warning("HF_TOKEN environment variable not set. Skipping login.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PodcastPipeline")
logger.info(f"🚀 Pipeline initialized on device: {device}")

def cleanup_gpu():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.info("🧹 GPU Memory Cleared.")

def cleanup_old_data():
    """Removes all files in data/ and outputs/ directories to ensure a fresh start."""
    dirs_to_clean = ["data/audio", "data/avatars", "data/animations", "outputs"]
    logger.info("🧹 Clearing old session data...")
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            try:
                # Delete the entire directory and recreate it
                shutil.rmtree(d)
                os.makedirs(d)
                logger.info(f"✅ Cleared and reset: {d}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to reset {d}. Reason: {e}")
        else:
            os.makedirs(d, exist_ok=True)

def run_pipeline(topic, cleanup=False):
    logger.info("Starting AI Podcast Generation Pipeline")
    if cleanup:
        cleanup_old_data()
    else:
        # Ensure directories exist even if not cleaning
        for d in ["data/audio", "data/avatars", "data/animations", "outputs"]:
            os.makedirs(d, exist_ok=True)

    try:
        # Step 1: Script Generation
        script = generate_podcast_script(topic)
        if not script:
            raise ValueError("Script generation failed.")

        # Step 2: Avatar Generation
        avatar_paths = generate_avatars()

        # Cleanup after heavy FLUX model
        cleanup_gpu()

        # Step 3: TTS Generation
        audio_data = generate_audio_files(script)

        # Step 4: Animation Generation
        animation_data = animate_avatars(audio_data, avatar_paths)

        # Cleanup after heavy SadTalker model
        cleanup_gpu()

        # Step 5: Final Montage
        final_video = create_montage(animation_data, avatar_paths)
        if final_video:
            logger.info(f"Pipeline completed successfully! Final video: {final_video}")
        else:
            logger.error("Pipeline failed at the montage step.")
            
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Podcast Generator")
    parser.add_argument("--topic", type=str, default="The role of Generative AI in Industrial Manufacturing", 
                        help="The topic for the podcast")
    parser.add_argument("--cleanup", action="store_true", help="Delete all old data before running")
    args = parser.parse_args()
    
    run_pipeline(args.topic, args.cleanup)
