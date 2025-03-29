import subprocess
import os

def convert_video_to_h264(input_path, output_path=None):
    """
    Convert video to H.264 codec in MP4 container to ensure compatibility with Streamlit.
    If output_path is not provided, creates a temporary file and replaces the original.
    """
    
    # If no output_path specified, create a temp file and replace original after
    if output_path is None:
        output_path = input_path + ".temp.mp4"
        replace_original = True
    else:
        replace_original = False
    
    # Command to convert to H.264 with compatible settings
    cmd = [
        'ffmpeg',
        '-y',  # Automatically overwrite
        '-i', input_path,
        '-c:v', 'libx264',  # H.264 codec
        '-profile:v', 'high',  # High profile (matches working video)
        '-preset', 'medium',
        '-vf', 'format=yuv420p',  # Ensure yuv420p color space
        '-c:a', 'aac',  # AAC audio
        '-b:a', '130k',  # Similar audio bitrate to working video
        '-movflags', '+faststart',  # Optimize for web playback
        output_path
    ]
    
    # Run the conversion
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    
    # If conversion was successful and we're replacing the original
    if result.returncode == 0 and replace_original:
        # Backup original file
        backup_path = input_path + ".backup"
        os.rename(input_path, backup_path)
        
        # Replace with converted file
        os.rename(output_path, input_path)
        
        # Remove backup if everything worked
        os.remove(backup_path)
        
        return input_path
    elif result.returncode == 0:
        return output_path
    else:
        # If conversion failed, return error message
        error_msg = result.stderr.decode('utf-8')
        print(f"Conversion failed: {error_msg}")
        return None