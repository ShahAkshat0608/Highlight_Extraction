import streamlit as st
from Gemini_Inference.download_yt_video import *
import yaml
import os

# Open config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    st.title("YouTube Video Short Creator")
    
    # Dropdown for series selection
    series_options = ["Kapil_Sharma", "Singing_Superstars", "Indian_Idol"]
    selected_series = st.selectbox("Select a Series:", series_options)

    # Input for YouTube video link
    youtube_link = st.text_input("Enter YouTube Video Link:")
    

    if youtube_link:
        episode_id = youtube_link.split('=')[-1]
        print(episode_id)
        episode_folder = f"{config['root_folder']}/{episode_id}"
        
        if not os.path.exists(episode_folder):
            os.makedirs(episode_folder)
            download_youtube_video(youtube_link)
        
            with st.expander("Show logs"):
                st.text("Downloading the video from YouTube...")
        
        create_temp_config(episode_id, selected_series)
        
        if os.path.exists(f"{episode_folder}/{episode_id}.mp4"):
            st.success("Video downloaded successfully!")
            
            # Store episode_id in session state
            st.session_state["episode_id"] = episode_id
            st.session_state["selected_series"] = selected_series
            
            st.divider()
            st.page_link("./pages/1_Temporal_detection.py", label="Go to Next Page", icon="⌛", use_container_width=True)
    

if __name__ == "__main__":
    main()