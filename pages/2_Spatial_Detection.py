import streamlit as st
import yaml
import os
import sys
import io
from ViNet_Saliency.extract_frames import get_frames
from SpatialLocalisation.saliency_tracking_pipeline_with_stabilisation import saliency_tracking_stabilisation
from utils.utils import *
import json
from pprint import pprint



def main():
    st.title("Spatial Localisation")
    
    # logger = StreamlitLogger()
    # sys.stdout = logger  # Redirect standard output to logger
    

    # Retrieve the latest episode ID
    episode_id = st.session_state.get("episode_id", None)
    
    if episode_id:
        
        # Open config_temp.yaml
        with open("config_temp.yaml", "r") as file:
            config = yaml.safe_load(file)
            
        episode_folder = f"{config['root_folder']}/{episode_id}"
        video_path = f"{episode_folder}/{episode_id}.mp4"


        st.text("1. Extract frames from the previous created temporal shorts for further processing for saliency prediction in those frames.")
        frames_output_path = f"{episode_folder}/frames"
        os.makedirs(frames_output_path, exist_ok=True)
        
        saliency_output_path = f"{episode_folder}/saliency_results"
        os.makedirs(saliency_output_path, exist_ok=True)


        if st.button("Extract Frames from Segments"):
            if len(os.listdir(frames_output_path)) != config['num_shorts'] and len(os.listdir(saliency_output_path)) != config['num_shorts']:
                print(f"Extracting frames for {video_path}")
                os.system(f'rm -rf {frames_output_path}/*')
                os.system(f'bash ./ViNet_Saliency/extract_frames.sh {config["root_folder"]}/{config["episode_id"]}')
                print("Extracted frames successfully.")
            elif len(os.listdir(frames_output_path)) == 0 and len(os.listdir(saliency_output_path)) == config['num_shorts']:
                st.success(" Saliency Frames already exist, no need to extract frames")
            elif len(os.listdir(frames_output_path)) == config['num_shorts']:
                st.success("Frames already exist")        
            
        st.divider()
        
        
        
        st.text("2. Get the saliency predictions of which parts are more important that the others spatially in the frame and save them.")    
        if st.button("Get Saliency of the video (frames)"):
            saliency_output_path = f"{episode_folder}/saliency_results"
            os.makedirs(saliency_output_path, exist_ok=True)
            if len(os.listdir(saliency_output_path)) != config['num_shorts']:
                if len(os.listdir(frames_output_path)) == config['num_shorts']:
                    print(f"Getting saliency for {video_path}")
                    
                    # delete all folders inside the saliency_output_path
                    os.system(f'rm -rf {saliency_output_path}/*')
                    
                    os.system("python3 ViNet_Saliency/generate_result.py") # main saliency code
                    print("Got saliency successfully.")
                    if len(os.listdir(saliency_output_path)) == config['num_shorts']:
                        st.success("Saliency frames created successfully!")
               
                else:
                    st.warning("Please extract frames first.")
               
                if len(os.listdir(saliency_output_path)) == config['num_shorts']:
                    # remove the content from the frames folder
                    os.system(f'rm -rf {frames_output_path}/*')
            else:
                print("Saliency already exists")
                st.success("Saliency frames already exist")
                    
                
        st.divider()
    
    
    
        st.text("3. Find the characters in the frames either by using 'face' or 'body' and choose the more important part of the video by combining these predictions with the above predictions from the saliency algorithm to get the final spatial chosen part.")
        col1,col2 = st.columns(2)
        
        
        
        option = "Face"
        with col1:
            option = st.radio("Choose the type of tracking:", ["Face", "Body"])
        
        with col2:
            if st.button("Saliency Tracking with Stabilisation"):
                final_output_path = f"{episode_folder}/final_output_{option}/"
                os.makedirs(final_output_path, exist_ok=True)
                tracking_folder = f"{episode_folder}/tracking_output_{option}"
                os.makedirs(tracking_folder, exist_ok=True)
                if len(os.listdir(tracking_folder)) != config['num_shorts']:
                    print("Performing Saliency Tracking with Stabilisation...")
                    os.system(f"rm -rf {episode_folder}/tracking_output_{option}/*")
                    os.system(f"rm -rf {episode_folder}/final_output_{option}/*")
                    os.system(f"rm -rf {episode_folder}/stabilised_output_{option}/*")
                    os.system(f"rm -rf {episode_folder}/salient_output_{option}/*")

    
                    saliency_tracking_stabilisation(config, detection_type=option) 
                    
                    # final_output_path = f"{episode_folder}/final_output_{option}/"
                    # # convert the generated shorts to mp4 h.264 format using ffmpeg
                    # for file in os.listdir(final_output_path):    
                    #     # os.system(f"ffmpeg -i {final_output_path}/{file} -c:v libx264 -crf 23 {tracking_folder}/{file.split('.')[0]}.mp4")
                    #     convert_video_to_h264(f"{final_output_path}/{file}", f"{tracking_folder}/{file}")
                    if len(os.listdir(tracking_folder)) == config['num_shorts']:
                        st.success("Saliency Tracking and Short Creation completed successfully.")
                        print("Saliency Tracking and Short Creation completed successfully.")
                
                else:                
                    print("Already done with inference beforehand sometime ago")
                
        st.divider()
        
        # check if the folder exists
        if os.path.exists(f"{episode_folder}/final_output_{option}/"): 
            if len(os.listdir(f"{episode_folder}/final_output_{option}/")) == config['num_shorts']:
                st.success("Shorts created successfully!")


            # zip the final_output folder
            os.system(f"zip -j {episode_folder}/final_output_{option}.zip {episode_folder}/final_output_{option}/*")

            file_path = f"{episode_folder}/final_output_{option}.zip"
            
            
            # Check if file exists
            if os.path.exists(file_path):
                # Open and read the file
                with open(file_path, "rb") as file:
                    file_content = file.read()
                
                # Create the download button with the actual file content
                st.download_button(
                    label="Download Generated Shorts",
                    data=file_content,
                    file_name=f"{episode_id}_generated_shorts_{option}.zip",
                    mime="application/zip"
                )
        
            #show all the videos inside the folder
            with st.expander("Show Generated Shorts"):
                videos = os.listdir(f"{episode_folder}/final_output_{option}/")
                st.markdown("The following videos were created:")
                for video in videos:
                    print(video)
                    st.video(f"{episode_folder}/final_output_{option}/{video}")
                    
    else:
        st.warning("No episode ID found. Please go back and enter a YouTube link.")
      
    #         st.page_link("./pages/Spatial_Detection.py", label="Get the Saliency of these Segments")
    
    
    # else:
    #     st.warning("No episode ID found. Please go back and enter a YouTube link.")

    # Button to go back
    st.text("")
    st.text("")
    st.text("")
    st.page_link("./pages/1_Temporal_detection.py", label="Back to Previous Page", icon="⌛")  # Change "app.py" to your main Streamlit file name

    # with st.expander("Show logs"):
    #     st.text(logger.get_output())  # Display logged output

if __name__ == "__main__":
    main()