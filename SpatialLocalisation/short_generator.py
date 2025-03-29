'''This file contains short generator class containing methods for all the steps of the short generation process.

It should be initialised for 1 video and then the generate_short method should be called to generate the short.
'''

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

def adjust_aspect_ratio(box, frame_height , frame_width , aspect_ratio=9/16):
    """Adjust bounding box to maintain a given aspect ratio."""
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2, (y1 + y2) / 2)

    # take the whole frame height and adjust width
    width = int(frame_height * aspect_ratio)
    half_width = width // 2

    x1 = max(0, center[0] - half_width)
    x2 = min(frame_width, center[0] + half_width)
    y1 = 0
    y2 = frame_height

    return int(x1), int(y1), int(x2), int(y2)

def iou(boxA , boxB):
    '''Compute IOU between 2 boxes in the format [x1, y1, x2, y2]'''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / float(boxA_area + boxB_area - inter_area)

def videoIOU(boxlist1 , boxlist2 , total_frames):
    total_iou = 0
    frames_in_boxlist1 = boxlist1.keys()
    frames_in_boxlist2 = boxlist2.keys()
    common_frames = set(frames_in_boxlist1).intersection(set(frames_in_boxlist2))
    for frame in common_frames:
        total_iou += iou(boxlist1[frame] , boxlist2[frame])

    return total_iou / total_frames

fixed_box_ratios = {
    "left_ratio": 0.3571,
    "right_ratio": 0.6429,
    "top_ratio": 0.05,
    "bottom_ratio": 0.95
}

class ShortGenerator:
    def __init__(self , yolo_model_path , video_path , saliency_maps_dir , tracking_output_video_path , salient_output_video_path ,  stabilised_output_video_path ,  final_output_video_path , dets_thresh = 0.6, iou_thresh=0.1):
        self.detection_model = YOLO(yolo_model_path)
        self.video_path = video_path
        self.tracking_output_video_path = tracking_output_video_path
        
        self.dets_thresh = dets_thresh
        self.iou_thresh = iou_thresh
        
        self.saliency_maps_dir = saliency_maps_dir
        self.salient_output_video_path = salient_output_video_path
        self.stabilised_output_video_path = stabilised_output_video_path
        self.final_output_video_path = final_output_video_path
        

        assert os.path.exists(video_path) , "Video path does not exist"
        assert os.path.exists(yolo_model_path) , "Yolo model path does not exist"
        assert os.path.exists(saliency_maps_dir) , "Saliency map directory does not exist"
    
    
    def generate_short(self):
        self.video_details = self.get_video_details(self.video_path)
        self.fixed_box = (
            int(fixed_box_ratios["left_ratio"] * self.video_details['frame_width']),
            int(fixed_box_ratios["top_ratio"] * self.video_details['frame_height']),
            int(fixed_box_ratios["right_ratio"] * self.video_details['frame_width']),
            int(fixed_box_ratios["bottom_ratio"] * self.video_details['frame_height']),
        )

        # Get shots
        self.cuts = self.get_shots(self.video_path)

        # Get tracking results
        tracking_results_save_path = self.tracking_output_video_path.split(".")[0] + ".json"
        if not os.path.exists(tracking_results_save_path):
            self.tracking_results = self.get_tracking_resuls(self.video_path)
            with open(tracking_results_save_path , "w") as f:
                json.dump(self.tracking_results , f)
        
        with open(tracking_results_save_path , "r") as f:
            self.tracking_results = json.load(f)
        
        # Get optimal boxes
        optimal_boxes_save_path = self.salient_output_video_path.split(".")[0] + ".json"
        if not os.path.exists(optimal_boxes_save_path):
            self.optimal_boxes = self.get_optimal_boxes()
            with open(optimal_boxes_save_path , "w") as f:
                json.dump(self.optimal_boxes , f)
        
        with open(optimal_boxes_save_path , "r") as f:
            self.optimal_boxes = json.load(f)

        # Stabilise the video
        self.stabilised_boxes = self.get_stabilised_boxes()

        # Generate the final "short"
        cap = cv2.VideoCapture(self.video_path)
        short_height = self.video_details['frame_height']
        short_width = int(self.video_details['frame_height'] * 9/16)

        out = cv2.VideoWriter(self.final_output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.video_details['fps'], (short_width, short_height))
        frame_no = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            stabilised_box = self.stabilised_boxes[frame_no]
            x1, y1, x2, y2 = int(stabilised_box[0]), int(stabilised_box[1]), int(stabilised_box[2]), int(stabilised_box[3])

            short_frame = frame.copy()
            short_frame = short_frame[y1:y2, x1:x2]
            short_frame = cv2.resize(short_frame, (short_width, short_height))
            out.write(short_frame)
            frame_no += 1

        cap.release()
        out.release()
        
        
        # add audio to the output video using audio from the video in self.video_path using ffmpeg and give the output as the final_output_video_path with audio from self.video_path
        os.system(f"ffmpeg -i {self.video_path} -i {self.final_output_video_path} -map 1:v:0 -map 0:a:0 -c:v copy -shortest {self.final_output_video_path.split('.')[0]}_audio.mp4")
        os.system(f"mv {self.final_output_video_path.split('.')[0]}_audio.mp4 {self.final_output_video_path}")
        

    def get_video_details(self , video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"fps":fps , "frame_count":frame_count , "frame_width":frame_width , "frame_height":frame_height}
    
    def get_shots(self , video_path):
        shot_list = detect(video_path, ContentDetector())
        cuts = []
        for i, shot in enumerate(shot_list):
            cuts.append(shot[0].get_frames())
        cuts.append(self.video_details['frame_count'])
        return cuts

    def get_tracking_resuls(self , video_path):
        frame_no = 0
        curr_cut = 0
        tracker = None
        tracking_results = {}
        out = cv2.VideoWriter(self.tracking_output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.video_details['fps'], (self.video_details['frame_width'], self.video_details['frame_height']))

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret , frame = cap.read()
            if not ret:
                break
            
            if frame_no == self.cuts[curr_cut]:
                # initialise the tracker
                tracker = OCSort(det_thresh = self.dets_thresh , max_age=60)
                tracking_results[curr_cut] = {}
                curr_cut += 1
            
            if tracker is not None:
                detections = self.detection_model(frame)
                # orgainse the detections in the format required by the tracker
                # [x1, y1, x2, y2, conf]
                tracker_input = []
                for detection in detections:
                    for box in detection.boxes:
                        if int(box.cls[0]) == 0: # only track persons
                            x1, y1, x2, y2 = map(int , box.xyxy[0])
                            # adjust the aspect ratio of the box
                            conf = float(box.conf[0])
                            tracker_input.append([x1, y1, x2, y2, conf])

                if len(tracker_input) == 0: # if no detections 
                    tracker_output = tracker.update(None , [self.video_details['frame_width'], self.video_details['frame_height']] , [self.video_details['frame_width'], self.video_details['frame_height']])
                    tracker_output = tracker_output.tolist()
                    tracking_results[curr_cut-1][frame_no] = tracker_output
                    out.write(frame)
                    frame_no += 1
                    continue

                tracker_input = np.array(tracker_input)                
                tracker_output = tracker.update(tracker_input , [self.video_details['frame_width'], self.video_details['frame_height']] , [self.video_details['frame_width'], self.video_details['frame_height']])
                tracker_output = tracker_output.tolist()
                # tracker_output is a list of lists containing the updated bounding boxes in the format [x1, y1, x2, y2, id]
                # adjust the aspect ratio of the boxes in tracker output
                for i in range(len(tracker_output)):
                    x1 , y1 , x2 , y2 , id = map(int , tracker_output[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    adjx1 , adjy1 , adjx2 , adjy2 = adjust_aspect_ratio([x1, y1, x2, y2] , self.video_details['frame_height'] , self.video_details['frame_width'])
                    tracker_output[i] = [adjx1, adjy1, adjx2, adjy2, id]

                tracking_results[curr_cut-1][frame_no] = tracker_output
                out.write(frame)
            
            frame_no += 1

        cap.release()
        out.release()
        return tracking_results
    
    def get_optimal_boxes(self):
        num_shots = len(self.tracking_results)
        optimal_boxes = {}
        for shot_no in range(num_shots):
            shot_tracking_results = self.tracking_results[str(shot_no)]
            num_frames_in_shot = len(shot_tracking_results)

            person_trajectories = {}
            for frame_no in range(num_frames_in_shot):
                actual_frame_no = frame_no + self.cuts[shot_no]
                for box in shot_tracking_results[str(actual_frame_no)]:
                    x1, y1, x2, y2, id = map(int, box)
                    if str(id) not in person_trajectories:
                        person_trajectories[str(id)] = {}
                    person_trajectories[str(id)][frame_no] = [x1, y1, x2, y2]
                
            person_ids = list(person_trajectories.keys())
            for id1 in person_ids:
                for id2 in person_ids:
                    if (id1 == id2):
                        continue
                    
                    person1_tracks = person_trajectories[id1]
                    person2_tracks = person_trajectories[id2]
                    avg_iou = videoIOU(person1_tracks, person2_tracks , num_frames_in_shot)
                   
                    if avg_iou > self.iou_thresh:
                        new_id = id1 + "_" + id2
                        person_trajectories[new_id] = {}
                        for frame_no in range(num_frames_in_shot):
                            if frame_no not in person_trajectories[id1].keys() and frame_no not in person_trajectories[id2].keys():
                                continue
                            if frame_no not in person_trajectories[id1].keys():
                                person_trajectories[new_id][frame_no] = person_trajectories[id2][frame_no]
                                continue
                            if frame_no not in person_trajectories[id2].keys():
                                person_trajectories[new_id][frame_no] = person_trajectories[id1][frame_no]
                                continue
                            x1 = (person_trajectories[id1][frame_no][0] + person_trajectories[id2][frame_no][0]) // 2
                            y1 = (person_trajectories[id1][frame_no][1] + person_trajectories[id2][frame_no][1]) // 2
                            x2 = (person_trajectories[id1][frame_no][2] + person_trajectories[id2][frame_no][2]) // 2
                            y2 = (person_trajectories[id1][frame_no][3] + person_trajectories[id2][frame_no][3]) // 2
                            person_trajectories[new_id][frame_no] = [x1, y1, x2, y2]
            
            saliency_scores = {}
            for person_id in person_trajectories.keys() :
                saliency_scores[person_id] = np.zeros(num_frames_in_shot)
        
            for frame_no in range(num_frames_in_shot):
            # saliency_map_path = # frame_1398.png , frame_0639.png etc i.e zfill(4)
                actual_frame_no = frame_no + self.cuts[shot_no]
                saliency_map_path = f"{self.saliency_maps_dir}/frame_{str(actual_frame_no+1).zfill(4)}.png" # frame nos are 1 indexed
                saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
                if saliency_map is None:
                    print(f"Error reading saliency map {saliency_map_path}")
                    continue
                    
                salient_mask = saliency_map * (saliency_map > 200)  # Retains pixel values for weights
                total_weighted_saliency = np.sum(salient_mask)
                for person_id in person_trajectories.keys():
                    if frame_no in person_trajectories[person_id].keys():
                        x1, y1, x2, y2 = person_trajectories[person_id][frame_no]
                        adjx1 , adjy1 , adjx2 , adjy2 = adjust_aspect_ratio([x1, y1, x2, y2] , self.video_details['frame_height'] , self.video_details['frame_width'])
                        person_mask = salient_mask[adjy1:adjy2, adjx1:adjx2]
                        saliency_scores[person_id][frame_no] = np.sum(person_mask) / total_weighted_saliency
            
            for person_id in saliency_scores.keys():
                saliency_scores[person_id] = saliency_scores[person_id].tolist()

            # Let us now try to do greedy optimisation at the person level , i.e choose the person which has the highest sum of saliency scores over all frames
            best_person = None
            best_score = 0
            for person_id in saliency_scores.keys():
                score = np.sum(saliency_scores[person_id])
                if score > best_score:
                    best_score = score
                    best_person = person_id
                
            for frame_no in range(num_frames_in_shot):
                actual_frame_no = frame_no + self.cuts[shot_no]
                if best_person is not None:
                    if frame_no not in person_trajectories[best_person].keys():
                        # get the previous frame box which is closest to the current frame
                        prev_frame_no = frame_no - 1
                        while prev_frame_no >= 0 and prev_frame_no not in person_trajectories[best_person].keys():
                            prev_frame_no -= 1
                        if prev_frame_no == -1:
                            optimal_boxes[actual_frame_no] = [adjust_aspect_ratio(self.fixed_box , self.video_details['frame_height'] , self.video_details['frame_width']),0]
                            continue
                        else:
                            optimal_boxes[actual_frame_no] = [adjust_aspect_ratio(person_trajectories[best_person][prev_frame_no] , self.video_details['frame_height'] , self.video_details['frame_width']) , best_score]
                    else:
                        optimal_boxes[actual_frame_no] = [adjust_aspect_ratio(person_trajectories[best_person][frame_no] , self.video_details['frame_height'] , self.video_details['frame_width']) , best_score]
                else:
                    optimal_boxes[actual_frame_no] = [adjust_aspect_ratio(self.fixed_box , self.video_details['frame_height'] , self.video_details['frame_width']),0]
            
           

        # Make the salient output video
        cap = cv2.VideoCapture(self.video_path)
        out = cv2.VideoWriter(self.salient_output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.video_details['fps'], (self.video_details['frame_width'], self.video_details['frame_height']))
        frame_no = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_no in optimal_boxes:
                [x1, y1, x2, y2] , score = optimal_boxes[frame_no]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(score), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str(frame_no), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            saliency_map_path = f"{self.saliency_maps_dir}/frame_{str(frame_no+1).zfill(4)}.png" # frame nos are 1 indexed
            saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
            # overlay the saliency map on the frame in rainbow colours
            heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_RAINBOW)
            heatmap = cv2.resize(heatmap, (self.video_details['frame_width'], self.video_details['frame_height']))
            frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
            

            out.write(frame)
            frame_no += 1

        cap.release()
        out.release()        
        return optimal_boxes
    
    def get_stabilised_boxes(self):
        optimal_boxes = self.optimal_boxes
        optimal_boxes = {int(k):v[0] for k, v in optimal_boxes.items()}
        optimal_boxes = optimal_boxes.values()
        optimal_boxes = list(optimal_boxes)

        stabilised_boxes = []
        for i in range(1, len(self.cuts)):
            if self.cuts[i] - self.cuts[i-1] < 2:
                stabilised_boxes.extend(optimal_boxes[self.cuts[i-1]:self.cuts[i]])
                continue
            
            
            stabilisation_manager = StabilisationManager(self.cuts[i] - self.cuts[i-1] , stabilise_in_degrees=3, frame_size=(self.video_details['frame_height'], self.video_details['frame_width']) , use_constraint_box=False )
            optimal_boxes_stabilised = stabilisation_manager.stabilise_rushes(optimal_boxes[self.cuts[i-1]:self.cuts[i]], [], lambdax1=100, lambdax2=100, lambday1=100, lambday2=100 , stabilise_x_y_together=False, frame_height=self.video_details['frame_height'])
            
            stabilised_boxes.extend(optimal_boxes_stabilised)

        cap = cv2.VideoCapture(self.video_path)
        out = cv2.VideoWriter(self.stabilised_output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.video_details['fps'], (self.video_details['frame_width'], self.video_details['frame_height']))
        frame_no = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            stabilised_box = stabilised_boxes[frame_no]
            x1, y1, x2, y2 = int(stabilised_box[0]), int(stabilised_box[1]), int(stabilised_box[2]), int(stabilised_box[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Selected Person"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            out.write(frame)
            frame_no += 1

        cap.release()
        out.release()

        return stabilised_boxes
