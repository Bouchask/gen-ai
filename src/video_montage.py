from moviepy import VideoFileClip, ImageClip, clips_array, concatenate_videoclips, CompositeVideoClip, ColorClip
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_montage(animation_data, avatar_paths, output_file="outputs/final_podcast.mp4"):
    """
    Stitches clips together into a side-by-side podcast layout and upscales to 1080p.
    Optimized for high-quality compression and visual professionality.
    """
    logging.info("🎬 Assembling final video in 1080p...")
    
    final_clips = []
    
    # Load static images and resize them once to match 512x512
    host_img = ImageClip(avatar_paths["Host"]).resized(height=512)
    expert_img = ImageClip(avatar_paths["Expert"]).resized(height=512)
    
    for item in animation_data:
        speaker = item['speaker']
        video_path = item['video_path']
        
        if not os.path.exists(video_path):
            logging.error(f"Video clip not found: {video_path}. Skipping.")
            continue
            
        speaking_clip = VideoFileClip(video_path)
        duration = speaking_clip.duration
        
        # Determine layout (Side-by-side 512x512 clips)
        if speaker == "Host":
            # Host speaks (left), Expert listens (right)
            listener_clip = expert_img.with_duration(duration)
            row = clips_array([[speaking_clip, listener_clip]])
        else:
            # Expert speaks (right), Host listens (left)
            listener_clip = host_img.with_duration(duration)
            row = clips_array([[listener_clip, speaking_clip]])
            
        final_clips.append(row)
        
    if not final_clips:
        logging.error("No clips to assemble!")
        return None
        
    # Concatenate all dialogue segments
    final_video = concatenate_videoclips(final_clips, method="compose")
    
    # Upscale and add a professional background
    logging.info("💎 Upscaling to 1920x1080 with high-fidelity encoding...")
    
    # Resize width to 1920 (height becomes 960)
    final_video_1080p = final_video.resized(width=1920)
    
    # Add to a 1080p canvas with a dark grey professional background
    # MoviePy 2.x: use CompositeVideoClip instead of on_color
    bg = ColorClip(size=(1920, 1080), color=(20, 20, 25)).with_duration(final_video_1080p.duration)
    final_video_1080p = CompositeVideoClip([bg, final_video_1080p.with_position("center")])

    logging.info(f"💾 Exporting to {output_file}...")
    
    # Using CRF 18 for visually lossless quality with better compression than fixed bitrate
    final_video_1080p.write_videofile(
        output_file, 
        fps=25, # Match SadTalker's native 25fps for perfect lip-sync
        codec="libx264", 
        audio_codec="aac",
        temp_audiofile='temp-audio.m4a', 
        remove_temp=True,
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"], # Industry standard for high-qual compression
        preset="slow",
        threads=4,
        logger=None # Suppress internal moviepy progress bars for cleaner logs
    )
    
    return output_file

if __name__ == "__main__":
    # Test
    pass
