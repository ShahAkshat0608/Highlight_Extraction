import cv2
import torch
from ultralytics import YOLO
from scenedetect import detect, ContentDetector
from SpatialLocalisation.ocsort_tracker.ocsort import OCSort
import numpy as np
import json
from SpatialLocalisation.stabilisation import StabilisationManager
import os
import time
from SpatialLocalisation.short_generator import ShortGenerator
import yaml

def saliency_tracking_stabilisation(config, detection_type='Face'):
    
    input_dir = f"{config['root_folder']}/{config['episode_id']}"

    input_video_names = os.listdir(os.path.join(input_dir, "raw_videos"))
    input_video_names = [video_name.split(".")[0] for video_name in input_video_names if video_name.endswith(".mp4")]
    print("Processing videos:", input_video_names)


    tracking_dir = os.path.join(input_dir, f"tracking_output_{detection_type}")
    salient_dir = os.path.join(input_dir, f"salient_output_{detection_type}")
    stabilised_dir = os.path.join(input_dir, f"stabilised_output_{detection_type}")
    final_dir = os.path.join(input_dir, f"final_output_{detection_type}")

    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(salient_dir, exist_ok=True)
    os.makedirs(stabilised_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    if detection_type == "Face":
        yolo_model_path = "./SpatialLocalisation/face_yolov8n.pt"
    else:
        yolo_model_path = "./SpatialLocalisation/yolov8n.pt"
        

    for video_name in input_video_names:
        short_generator = ShortGenerator(
            yolo_model_path=yolo_model_path,
            video_path=os.path.join(input_dir, "raw_videos", f"{video_name}.mp4"),
            saliency_maps_dir=os.path.join(input_dir, "saliency_results", video_name),
            tracking_output_video_path=os.path.join(tracking_dir, f"{video_name}.mp4"),
            salient_output_video_path=os.path.join(salient_dir, f"{video_name}.mp4"),
            stabilised_output_video_path=os.path.join(stabilised_dir, f"{video_name}.mp4"),
            final_output_video_path=os.path.join(final_dir, f"{video_name}.mp4"),
            dets_thresh = config['detection_threshold'],
            iou_thresh = config['iou_threshold']
        )

        print(f"Generating short video for {video_name}...")
        short_generator.generate_short()

        print(f"Short video generated successfully for {video_name}!")