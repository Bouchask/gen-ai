import subprocess
import os
import logging
import glob
import shutil
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def animate_avatars(audio_data, avatar_paths, output_dir="data/animations"):
    """
    Wraps SadTalker batch script to animate avatars.
    Loads models once for all clips, providing massive speedup.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    abs_output_dir = os.path.abspath(output_dir)
    abs_sadtalker_dir = os.path.abspath("models/SadTalker")
    batch_script = "animate_batch.py"
    tasks_file = os.path.join(abs_output_dir, "tasks.json")
    
    if not os.path.exists(os.path.join(abs_sadtalker_dir, batch_script)):
        logging.error(f"Batch script not found at {os.path.join(abs_sadtalker_dir, batch_script)}")
        return []

    tasks = []
    animation_clips = []

    for item in audio_data:
        speaker = item['speaker']
        audio_path = os.path.abspath(item['audio_path'])
        img_path = os.path.abspath(avatar_paths.get(speaker))
        
        if not img_path or not os.path.exists(img_path):
            continue
            
        final_name = os.path.basename(audio_path).replace(".wav", ".mp4")
        final_path = os.path.join(abs_output_dir, final_name)
        
        if os.path.exists(final_path):
            logging.info(f"⏭️ Animation for {os.path.basename(audio_path)} already exists.")
            animation_clips.append({
                "speaker": speaker, 
                "video_path": final_path, 
                "audio_path": audio_path
            })
            continue

        # Host is more expressive, Expert is more composed
        expr_scale = 1.6 if speaker == "Host" else 1.4
        
        tasks.append({
            "audio": audio_path,
            "image": img_path,
            "output": final_path,
            "expression_scale": expr_scale,
            "still": True # Keeping still=True for frontal portraits to avoid neck artifacts
        })

    if not tasks:
        return animation_clips

    # Write tasks to JSON
    with open(tasks_file, 'w') as f:
        json.dump(tasks, f)

    logging.info(f"🎬 Starting high-fidelity batch animation for {len(tasks)} clips...")
    
    cmd = [
        "python", batch_script,
        "--tasks_json", tasks_file,
        "--result_dir", abs_output_dir,
        "--device", "cuda",
        "--fp16", # Enable FP16 for faster rendering and better memory management
        "--preprocess", "full",
        "--enhancer", "gfpgan",
        "--background_enhancer", "realesrgan",
        "--expression_scale", "1.3", # Global fallback
        "--size", "512" 
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=abs_sadtalker_dir)
        
        # Verify all outputs
        for task in tasks:
            if os.path.exists(task['output']):
                # Find matching speaker from audio_data
                speaker = next(item['speaker'] for item in audio_data if os.path.abspath(item['audio_path']) == task['audio'])
                animation_clips.append({
                    "speaker": speaker, 
                    "video_path": task['output'], 
                    "audio_path": task['audio']
                })
        
        logging.info("✅ Batch animation complete.")
    except Exception as e:
        logging.error(f"❌ Error during batch animation: {e}")
    finally:
        if os.path.exists(tasks_file):
            os.remove(tasks_file)
            
    return animation_clips
