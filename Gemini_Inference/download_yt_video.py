# %%
import subprocess
import yaml


# open config_temp.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
path=f"{config['root_folder']}/"

def download_youtube_video(url, custom_filename="custom_video_name.mp4"):
    """Downloads a YouTube video with around 480p quality and saves it with a custom name in MP4 format."""
    if url[len(url)-1]=='/':
        url=url[:len(url)-1]
    
    episode_id = url.split("=")[-1]    
    
    custom_filename= f"{config['root_folder']}/{episode_id}/{episode_id}.mp4"
    
    print(custom_filename)
    # yt-dlp command to download video around 480p quality
    command = [
        "yt-dlp",
        "-f", "bestvideo[height<=560]+bestaudio/best",  # Ensure 480p quality or lower, best audio
        "-o", f"{custom_filename}",        # Save with the custom filename
        "--merge-output-format", "mp4",    # Ensure the format is MP4
        url                               # The URL of the video
    ]
    
    try:
        # Run the command in the terminal
        subprocess.run(command, check=True)
        print(f"Download complete! File saved as {custom_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    
    return episode_id

def create_temp_config(episode_id, series):
    with open ("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    config['episode_id'] = episode_id
    config['series'] = series
    
    with open ("config_temp.yaml", "w") as file:
        yaml.dump(config, file)
        
    


