
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from IPython.display import Markdown
import concurrent.futures
import time
import subprocess
import yaml
import json
import ast
from moviepy.editor import VideoFileClip

def setup_gemini():
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    # print(gemini_api_key)
    return gemini_api_key
# print(config['episode_id'])

def get_timestamps(video_path, segment_length, overlap):
    duration_cmd = f'ffprobe -i "{video_path}" -show_entries format=duration -v quiet -of csv="p=0"'
    duration = float(subprocess.check_output(duration_cmd, shell=True).decode().strip())
    timestamps = [i for i in range(0, int(duration), segment_length - overlap)]
    return timestamps


def create_overlapping_segments(config, input_file, output_path, episode_id, timestamps, segment_length=420, overlap=120):
    
    os.makedirs(output_path, exist_ok=True)
    
    vid_name=input_file.split('.')[0]
    # print(vid_name)
    for i, start_time in enumerate(timestamps):
        output_file = f"{output_path}/{episode_id}_segment_{i:03d}.mp4"
        cmd = f'ffmpeg -i "{input_file}" -ss {start_time} -t {segment_length} -c copy "{output_file}"'
        subprocess.run(cmd, shell=True)


# Prompt with a video and text
def do_gemini_inference(config, gemini_api_key):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name=config['gemini_model'])

    output_path = f"{config['root_folder']}/{config['episode_id']}/input_temp_segments"

    if config['series']=='Kapil_Sharma':
        with open(config['Kapil_Sharma_prompt_file'], 'r') as f:
            prompt = f.read()
        
    elif config['series']=='Singing_Superstars':
        with open(config['Singing_Superstars_prompt_file'], 'r') as f:
            prompt = f.read()
            
    elif config['series']=='Indian_Idol':
        with open(config['Indian_Idol_prompt_file'], 'r') as f:
            prompt = f.read()


    duration = int(config['segment_length']/60)

    prompt = prompt + f"\nThe video is {duration} minutes long so make sure that the timestamps do not exceed that time limit"

    segments = sorted(os.listdir(output_path))
    

    def process_video(segment, output_path, config, model, prompt):
        video_path = f"{output_path}/{segment}"
        video_name = segment.split('.')[0]
        
        print(f"Uploading file {video_name}...")
        video_file = genai.upload_file(path=video_path)
        print(f"Completed upload: {video_file.uri}")
        
        while video_file.state.name == "PROCESSING":
            # print('.', end='')
            time.sleep(5)  # Reduced sleep time
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"File processing failed: {video_name}")
        
        print(f"Making LLM inference request for {video_name}...")
        response = model.generate_content(
            [video_file, prompt], 
            request_options={"timeout": 600},
            generation_config={"temperature": config['gemini_temperature']}
        )
        
        output_file = f"{config['root_folder']}/{config['episode_id']}/outputs_gemini/{video_name}_output.txt"
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        return f"Completed processing {video_name}"


    os.makedirs(f"{config['root_folder']}/{config['episode_id']}/outputs_gemini/",exist_ok=True)

    # Number of concurrent workers
    max_workers = config['workers_gemini']  # Adjust based on your system capabilities

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_video, segment, output_path, config, model, prompt) 
            for segment in segments
        ]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error during processing: {e}")



def combine_jsons(config):
    # ### Combine all to make one json
    output_path = f"{config['root_folder']}/{config['episode_id']}/input_temp_segments"
    segments = sorted(os.listdir(output_path))

    final_json = []

    def adjust_timestamp(timestamp, segment_number):
        """Adjusts the timestamp by adding 5 * segment_number minutes."""
        def to_seconds(hh_mm_ss):
            """Converts hh:mm:ss format to seconds."""
            hh, mm, ss = map(int, hh_mm_ss.split(':'))
            return hh * 3600 + mm * 60 + ss

        def to_hhmmss(seconds):
            """Converts seconds back to hh:mm:ss format."""
            hh = seconds // 3600
            mm = (seconds % 3600) // 60
            ss = seconds % 60
            return f"{hh:02}:{mm:02}:{ss:02}"

        # Split start and end timestamps
        start_str, end_str = timestamp.split('-')
        
        # Convert to seconds
        start_sec = to_seconds("00:"+start_str ) + (5 * segment_number * 60)
        end_sec = to_seconds("00:"+end_str) + (5 * segment_number * 60)
        
        # Convert back to hh:mm:ss
        new_start = to_hhmmss(start_sec)
        new_end = to_hhmmss(end_sec)
        
        duration = end_sec - start_sec

        return f"{new_start}-{new_end}", duration

    for segment in segments:
        video_name = segment.split('.')[0]
        segment_number = int(video_name.split('_')[-1])

        output_file = os.path.join(config['root_folder'], config['episode_id'], 'outputs_gemini', f"{video_name}_output.txt")
        
        with open(output_file, 'r') as f:
            output_text = f.read()
        
        # Extract JSON content properly
        try:
            # Find the start and end of the JSON array
            start_idx = output_text.find('[')
            end_idx = output_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_text = output_text[start_idx:end_idx]
                # Use json.loads instead of ast.literal_eval
                output = json.loads(json_text)
            else:
                print(f"Warning: Could not find JSON array in {output_file}")
                continue
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in file {output_file}: {e}")
            continue

        # Adjust timestamps and append to final_json
        for entry in output:
            entry['timestamp'], entry['duration'] = adjust_timestamp(entry['timestamp'], segment_number)
            final_json.append(entry)

    # save the json 
    output_file = os.path.join(config['root_folder'], config['episode_id'], f"{config['episode_id']}_timestamps.json")
    with open(output_file, 'w') as f:
        json.dump(final_json, f, indent=4)

    if config['series']=='Kapil_Sharma':
        filtered_timestamps = [entry for entry in final_json if (entry['duration']>=config['min_duration'] and entry['duration']<=config['max_duration'] and entry['laughter_intensity'] >= config['minimum_laughter_intensity'])]

    elif config['series']=='Singing_Superstars' or config['series']=='Indian_Idol':
        filtered_timestamps = [entry for entry in final_json if (entry['duration']>=config['min_duration'] and entry['duration']<=config['max_duration'] and entry['musical_intensity'] >= config['minimum_musical_intensity'] and entry['emotional_intensity'] >= config['minimum_emotional_intensity'] and entry['laughter_intensity'] >= config['minimum_laughter_intensity'])]
        
    # save the json 
    output_file = os.path.join(config['root_folder'], config['episode_id'], f"{config['episode_id']}_timestamps_filtered.json")
    with open(output_file, 'w') as f:
        json.dump(filtered_timestamps, f, indent=4)
        
    return final_json, filtered_timestamps    # # Create the videos from the filtered timestamps


def get_final_temporal_clips(config, filtered_timestamps, video_path):

    print('> Creating the final raw videos from the filtered timestamps')

    def extract_clip(input_video, start_time, duration, output_file):
        # Convert start_time (HH:MM:SS) to seconds
        h, m, s = map(float, start_time.split(':'))
        start_seconds = h * 3600 + m * 60 + s
        
        # Load the video file
        try:
            with VideoFileClip(input_video) as video:
                # Extract the subclip
                subclip = video.subclip(start_seconds, start_seconds + float(duration))
                # Write to output file
                subclip.write_videofile(
                    output_file,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
            print(f"Extracted: {output_file}")
        except Exception as e:
            print(f"Error extracting {output_file}: {e}")

    raw_videos_folder = f"{config['root_folder']}/{config['episode_id']}/raw_videos"
    os.makedirs(raw_videos_folder, exist_ok=True)

    # Process each timestamp
    for index, item in enumerate(filtered_timestamps):
        start_time, end_time = item["timestamp"].split("-")
        duration = item["duration"]
        category = item["category"].replace(" ", "_")  # Format category for filename
        output_filename = f"{raw_videos_folder}/{config['episode_id']}_{index+1}.mp4"
        
        extract_clip(video_path, start_time, duration, output_filename)

    print("All clips have been extracted successfully!")

