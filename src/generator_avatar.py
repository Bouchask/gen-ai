import torch
from diffusers import FluxPipeline
import logging
import os
import gc
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_gpu_memory():
    """Returns free VRAM in MB."""
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        free_vram = int(subprocess.check_output(command.split()).decode().strip())
        return free_vram
    except:
        return 0

def generate_avatars(output_dir="data/avatars"):
    """
    Optimized for 15GB RAM / T4 GPU.
    Uses SDXL-Turbo for speed and reliability.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    prompts = {
        "Host": "Photorealistic frontal portrait of a professional male podcast host looking directly into the camera, extreme detail on face and eyes, highly detailed skin texture, professional podcast studio with high-end microphone in foreground, cinematic lighting, 8k UHD, masterpiece, sharp focus",
        "Expert": "Photorealistic frontal portrait of a professional female expert looking directly into the camera, intelligent expression, extreme facial detail, sharp eyes, subsurface scattering, set in a modern high-tech podcast laboratory, cinematic lighting, 8k UHD, masterpiece"
    }

    avatar_paths = {}
    
    # Check existence
    all_exist = True
    for speaker in prompts:
        path = os.path.join(output_dir, f"{speaker.lower()}.png")
        if os.path.exists(path):
            avatar_paths[speaker] = path
            logging.info(f"⏭️ Avatar for {speaker} already exists.")
        else:
            all_exist = False
            
    if all_exist:
        return avatar_paths

    logging.info("🚀 Loading SDXL-Turbo (Optimized for 15GB RAM)...")
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Kill ollama to free up RAM if it's still running
    os.system("pkill -9 ollama > /dev/null 2>&1 || true")

    try:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16"
        )
        pipe.to("cuda")
        
        for speaker, prompt in prompts.items():
            path = os.path.join(output_dir, f"{speaker.lower()}.png")
            if os.path.exists(path):
                continue

            logging.info(f"🎨 Generating {speaker} (SDXL-Turbo)...")
            image = pipe(
                prompt=prompt, 
                num_inference_steps=4, # Increased for better facial detail
                guidance_scale=0.0, 
                width=512, 
                height=512
            ).images[0]
            
            image.save(path)
            avatar_paths[speaker] = path
            logging.info(f"✅ Saved {speaker} avatar to {path}")

        # Final Cleanup
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return avatar_paths

    except Exception as e:
        logging.error(f"❌ Avatar generation failed: {e}")
        return avatar_paths

if __name__ == "__main__":
    generate_avatars()
