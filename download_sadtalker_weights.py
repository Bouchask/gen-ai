import os
import requests
import logging

logging.basicConfig(level=logging.INFO)

def download_file(url, dest):
    if os.path.exists(dest):
        logging.info(f"⏭️ {dest} already exists.")
        return
    logging.info(f"📥 Downloading {url} to {dest}...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    base_url = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
    checkpoint_dir = "models/SadTalker/checkpoints"
    gfpgan_dir = "models/SadTalker/gfpgan/weights"
    
    files = {
        "auido2exp_00062-00109-00148.pth": checkpoint_dir,
        "auido2pose_00140-00144.pth": checkpoint_dir,
        "epoch_20.pth": checkpoint_dir,
        "facevid2vid_00189-00120.pth": checkpoint_dir,
        "mapping_00109-00231.pth": checkpoint_dir,
        "mapping_00229-00144.pth": checkpoint_dir,
        "shape_predictor_68_face_landmarks.dat": checkpoint_dir,
        "wav2lip.pth": checkpoint_dir,
        "GFPGANv1.4.pth": gfpgan_dir
    }
    
    for filename, folder in files.items():
        url = base_url + filename
        dest = os.path.join(folder, filename)
        download_file(url, dest)
        
    logging.info("✅ All SadTalker weights downloaded successfully!")

if __name__ == "__main__":
    main()
