import streamlit as st
import yaml
import os
import sys
import io
from Gemini_Inference.gemini_inference import *
from Gemini_Inference.graph_patches import *
import json
from pprint import pprint

# Open config_temp.yaml
with open("config_temp.yaml", "r") as file:
    config = yaml.safe_load(file)


def main():
    st.title("Temporal Segment Detection")
    
    # logger = StreamlitLogger()
    # sys.stdout = logger  # Redirect standard output to logger

    # Retrieve the latest episode ID
    episode_id = st.session_state.get("episode_id", None)
    
    if episode_id:
        episode_folder = f"{config['root_folder']}/{episode_id}"
        video_path = f"{episode_folder}/{episode_id}.mp4"

        # Setup Gemini API
        gemini_api_key = setup_gemini()
        print("Gemini API setup complete.")
        # print(gemini_api_key)

        # Create overlapping segments from the video
        st.text("1. Creating overlapping segments from the previous downloaded video of 7 minutes with 2 minutes overlapping between each segments to make sure context is missed.")
        
        timestamps = get_timestamps(video_path, config['segment_length'], config['overlap'])
        number_of_overlapping_segments = len(timestamps)
        if st.button("Create Overlapping Segments"):
            output_path = f"{episode_folder}/input_temp_segments"
            
            os.makedirs(output_path, exist_ok=True)
            if len(os.listdir(output_path)) != len(timestamps):
                print(f"Creating overlapping segments for {video_path}")
                create_overlapping_segments(config, video_path, output_path, episode_id, timestamps, config['segment_length'], config['overlap'])
                st.markdown(f"Number of overlapping segments created: {len(timestamps)}")
                st.success("Overlapping segments created successfully!")
                print("Overlapping segments created successfully.")
            else:
                print("Segments already exist")
                st.markdown(f"Number of already existing overlapping segments: {len(timestamps)}")
                st.success("Segments already exist!")
                
        st.divider()
        
        
        
    
        st.text("2. Doing inference on Gemini using Video + Prompt to get the important temporal parts from the overlapping segments created previously with a description of the segment")
       
        temperature = st.text_input("Enter custom temperature for Gemini", value='0')
        gemini_model = st.text_input("Enter model on which you want inference", value="gemini-1.5-pro")
        
        if temperature:
            temperature = int(temperature)
            if temperature>=0 and temperature<=1:
            # Update config_temp.yaml with custom temperature
                config['temperature'] = float(temperature)
                with open("config_temp.yaml", "w") as file:
                    yaml.dump(config, file)
                    
        if gemini_model:
            # Update config_temp.yaml with custom model
            config['gemini_model'] = gemini_model
            with open("config_temp.yaml", "w") as file:
                yaml.dump(config, file)
    
        os.makedirs(f"{episode_folder}/outputs_gemini", exist_ok=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Do Gemini Inference"):
                if len(os.listdir(f"{episode_folder}/outputs_gemini")) != number_of_overlapping_segments:
                    print("Performing Gemini Inference...")
                    do_gemini_inference(config, gemini_api_key)  
                    print("Gemini Inference completed successfully.")
                else:
                    print("Inference already done")
                    st.success("Inference already done!")
                    
        with col2:        
            # button to remove the inferences
            if st.button("Remove any Previous Inferences?"):
                
                os.system(f"rm -rf {episode_folder}/outputs_gemini/*")
                
                print("Previous inferences removed!")
                st.warning("Previous inferences removed!")

                
            
        st.divider()
            
            
            
        st.text("3. From the above created JSONs of individual segments, combine them to make one JSON containing all the important short-worthy segments along with their timestamps and the descriptions")
        # st.info('There might be an issue with the code where if the descriptions that were output by Gemini have a single qoute or a double qoute, then you will have to manually fix it in the folder `outputs_gemini` and remove them to run the Combine JSONs function')
        timestamps=None
        filtered_timestamps=None
        
        # input to get the min_duration, max_duration, min_score, max_score
        min_duration = st.text_input("Enter minimum duration for the shorts", value='10')
        max_duration = st.text_input("Enter maximum duration for the shorts", value='60')
        
        # update config_temp.yaml with the min_duration and max_duration
        config['min_duration'] = int(min_duration)
        config['max_duration'] = int(max_duration)
        
        if config['series'] == 'Kapil_Sharma':
            minimum_laughter_intensity = st.text_input("Enter minimum laughter intensity for the shorts (1-5)", value='5')
            config['minimum_laughter_intensity'] = int(minimum_laughter_intensity)
            
        elif config['series'] == 'Singing_Superstars' or config['series'] == 'Indian_Idol':
            minimum_musical_intensity = st.text_input("Enter minimum singing intensity for the shorts (1-5)", value='5')
            minimum_emotional_intensity = st.text_input("Enter minimum emotion intensity for the shorts (1-5)", value='5')
            minimum_laughter_intensity = st.text_input("Enter minimum laughter intensity for the shorts (1-5)", value='5')
            config['minimum_musical_intensity'] = int(minimum_musical_intensity)
            config['minimum_emotional_intensity'] = int(minimum_emotional_intensity)
            config['minimum_laughter_intensity'] = int(minimum_laughter_intensity)
            
        with open("config_temp.yaml", "w") as file:
            yaml.dump(config, file)
        
        
        if st.button("Combine JSONs"):
            # check if the outputs_gemini is empty or not
            if len(os.listdir(f"{episode_folder}/outputs_gemini")) == 0:
                st.warning("No inferences found. Please perform Gemini Inference first.")
            elif os.listdir(f"{episode_folder}/outputs_gemini") != number_of_overlapping_segments:
                timestamps, filtered_timestamps = combine_jsons(config)
                st.success("JSONs combined successfully!")
                st.code(f"Length of timestamps: {len(timestamps)}\nLength of filtered timestamps: {len(filtered_timestamps)}", language='Markdown')
                # st.markdown(f"Length of filtered timestamps: {len(filtered_timestamps)}")
                # update the config_temp with the number of segments
                config['num_shorts'] = len(filtered_timestamps)
                with open("config_temp.yaml", "w") as file:
                    yaml.dump(config, file)
            elif os.listdir(f"{episode_folder}/outputs_gemini") != number_of_overlapping_segments:
                st.warning("Inference not done on all the segments. Please perform Gemini Inference first.")
            
        col1, col2 = st.columns(2)
        
        # show timestamps and filtered_timestamps in the two columns 
        with col1:
            # provide a button to download the timestamps
            if os.path.exists(f'{episode_folder}/{episode_id}_timestamps.json'):
                with open(f'{episode_folder}/{episode_id}_timestamps.json', 'r') as f:
                    timestamps = json.load(f)
                # Create the download button
                timestamps = json.dumps(timestamps, ensure_ascii=False, indent=4)
                st.download_button(
                    label="Download timestamps json",
                    data=timestamps,
                    file_name=f"{episode_id}_timestamps.json",
                )
            
        with col2:                
            if os.path.exists(f'{episode_folder}/{episode_id}_timestamps_filtered.json'):
                with open(f'{episode_folder}/{episode_id}_timestamps_filtered.json', 'r') as f:
                    filtered_timestamps = json.load(f)
                filtered_timestamps = json.dumps(filtered_timestamps, ensure_ascii=False, indent=4)
                # Create the download button
                st.download_button(
                    label="Download filtered timestamps",
                    data=filtered_timestamps,
                    file_name=f"{episode_id}_timestamps_filtered.json",
                    mime="application/json"
                )
                    
        if os.path.exists(f'{episode_folder}/{episode_id}_timestamps_filtered.json'):
            # show the filtered timestamps in st.expander
            with st.expander("Show Filtered Timestamps"):
                if filtered_timestamps and len(filtered_timestamps) > 0:
                    st.markdown("The following are the filtered timestamps:")
                    st.code(filtered_timestamps, language='json')
                else:
                    st.warning("Filtered timestamps not found or are empty.")


        if os.path.exists(f'{episode_folder}/{episode_id}_timestamps_filtered.json'):
            with open(f'{episode_folder}/{episode_id}_timestamps_filtered.json', 'r') as f:
                filtered_timestamps = json.load(f)
            # create the graph for the filtered timestamps
            if len(filtered_timestamps) > 0:
                create_graph(episode_folder, episode_id, filtered_timestamps)
        
                # show the image created in streamlit
                st.image(f"{episode_folder}/{episode_id}_category_timeline.png", use_container_width=True)
            
        
        
        
        
        st.divider()
        
        
        
        
        
        st.text("4. Cut the episode using the timestamps from the filtered timestamps file and Create Temporal Cut Shorts from the filtered timestamps and save them for further spatial processing in the next part.")
        if st.button("Create Temporal Cut Shorts"):
            if len(os.listdir(f"{episode_folder}/raw_videos/")) != config['num_shorts']:
                
                # remove any contents inside the raw_videos folder
                os.system(f"{episode_folder}/raw_videos/*")
                
                # check if the filtered_timestamps json exists or not
                if os.path.exists(f'{episode_folder}/{episode_id}_timestamps_filtered.json'):                    
                    with open(f'{episode_folder}/{episode_id}_timestamps_filtered.json', 'r') as f:
                            filtered_timestamps = json.load(f)
                        
                    get_final_temporal_clips(config, filtered_timestamps, video_path)
                    # st.success("Temporal cut shorts created successfully!")
                    
                else:
                    st.warning("No filtered timestamps found. Please combine the JSONs first.")
            
            else:
                print("Shorts already exist")
                st.success("Temporal cut shorts already exist!")
        
        
        os.makedirs(f"{episode_folder}/raw_videos", exist_ok=True) 
        if len(os.listdir(f"{episode_folder}/raw_videos/")) == config['num_shorts']:
            st.success("Temporal cut shorts created successfully!")
          
            #show all the videos inside the folder
            with st.expander("Show Generated Videos"):
                videos = os.listdir(f"{episode_folder}/raw_videos/")
                st.markdown("The following videos were created:")
                for video in videos:
                    st.video(f"{episode_folder}/raw_videos/{video}")
                    
            st.divider()  
            st.text("")
            st.page_link("./pages/2_Spatial_Detection.py", label="Get the Saliency of these Segments", icon="🛸")
    
    
    else:
        st.warning("No episode ID found. Please go back and enter a YouTube link.")

    # Button to go back
    st.page_link("Home.py", label="Back to Main Page", icon="📄")  # Change "app.py" to your main Streamlit file name

    # with st.expander("Show logs"):
    #     st.text(logger.get_output())  # Display logged output

if __name__ == "__main__":
    main()