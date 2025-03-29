import subprocess
import os

def combine_videos(video1, video2, output):
    command = [
        "ffmpeg", "-i", video1, "-i", video2, "-filter_complex",
        # Resize both videos to have the same dimensions
        "[0:v][1:v]scale2ref=oh*mdar:ih[v1][v2]; "  # Match height, adjust width
        "[v1]scale=iw:ih[v1_scaled]; [v2]scale=iw:ih[v2_scaled]; "  # Ensure exact match

        # Draw text labels
        "[v1_scaled]drawtext=text='Youtube Short':x=10:y=H-th-10:fontsize=24:fontcolor=white[left]; "
        "[v2_scaled]drawtext=text='Generated Short':x=10:y=H-th-10:fontsize=24:fontcolor=white[right]; "

        # Stack videos horizontally
        "[left][right]hstack=inputs=2[v]; "

        # Use audio from the first video
        "[0:a]aformat=channel_layouts=stereo[a]",

        # Map outputs
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k", "-y", output
    ]
    
    subprocess.run(command, check=True)




generated_short_dir = 'SingingSuperstar/Dataset/Kem7q2lRHZ0/final_output'
youtube_short_dir = 'SingingSuperstar/Dataset/Kem7q2lRHZ0/original_shorts'
parallel_video_dir = 'SingingSuperstar/Dataset/Kem7q2lRHZ0/parallel_videos'
os.makedirs(parallel_video_dir, exist_ok=True)

episode_id = 'Kem7q2lRHZ0'

shot_ids = os.listdir(youtube_short_dir)
shot_ids = [shot_id.split('.')[0] for shot_id in shot_ids if shot_id.endswith('.mp4')]

for shot_id in shot_ids:
    generated_short = os.path.join(generated_short_dir, episode_id + '_' + shot_id + '.mp4')
    youtube_short = os.path.join(youtube_short_dir, shot_id + '.mp4')
    output = os.path.join(parallel_video_dir, episode_id + '_' + shot_id + '.mp4')
    
    try:
        combine_videos(youtube_short, generated_short, output)
    except Exception as e:
        print(f"Failed to combine videos for {shot_id}: {e}")
    else:
        print(f"Combined videos for {shot_id} successfully")