# %%
import os
import cv2
import torch
from model import VideoSaliencyModel
import argparse
from utils import *
from os.path import join
from torchvision import transforms
import yaml
from PIL import Image

import concurrent.futures
import concurrent.futures
import os
from os.path import join
import torch
from torch.utils.data import DataLoader , Dataset
from moviepy import VideoFileClip
import tqdm

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# suppress warnings and logs
import warnings
warnings.filterwarnings("ignore")

VIDEO_DIR = '/ssd_scratch/cvit/sarthak395/outputs/IyMdcXl4vag/raw_videos'
OUTPUT_DIR = '/ssd_scratch/cvit/sarthak395/outputs/IyMdcXl4vag/saliency_results_batching'
BATCH_SIZE = 8
FILE_WEIGHT = '/home2/sarthak395/Sony_Shorts_Creator/ViNet_Saliency/saved_models/ViNet_DHF1K.pt'

# %%
class SaliencyPredictionDataset(Dataset):
    def __init__(self , video_dir , save_dir , len_temporal = 32):
        self.video_dir = video_dir
        self.len_temporal = len_temporal
        self.save_dir = save_dir
        self.video_list = os.listdir(video_dir) # contains the video names : ['IyMdcXl4vag_1.mp4' , 'IyMdcXl4vag_2.mp4' , ...]
        self.video_list.sort()
        self._create_samples()
    
    def _create_samples(self):
        self.samples = [] # contains the samples of the form (video_path , start_frame , end_frame , save_path , flip)
        for video_name in self.video_list:
            os.makedirs(os.path.join(self.save_dir , video_name.split('.')[0]) , exist_ok=True)
            video_path = os.path.join(self.video_dir , video_name)
            # clip = VideoFileClip(video_path)# remove logs
            clip  = VideoFileClip(video_path)
            num_frames = int(clip.fps * clip.duration)
            
            # processing the initial (len_temporal-1) frames
            for i in range(self.len_temporal - 1):
                if i < num_frames:
                    start_frame = i
                    end_frame = i + self.len_temporal - 1 # total frames = len_temporal
                    save_path = os.path.join(self.save_dir , video_name.split('.')[0] , f'frame_{i:04d}.png')
                    self.samples.append((video_path , start_frame , end_frame , save_path , True)) # need to flip the frames
                else:
                    break

            # processing the rest of the frames
            for i in range(self.len_temporal - 1 , num_frames):
                start_frame = i - self.len_temporal + 1
                end_frame = i
                save_path = os.path.join(self.save_dir , video_name.split('.')[0] , f'frame_{i:04d}.png')
                self.samples.append((video_path , start_frame , end_frame , save_path , False))
            
        print(f'Total samples created : {len(self.samples)}')
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path , start_frame , end_frame , save_path , flip = self.samples[idx]
        clip = VideoFileClip(video_path)
        
        # get image size of the first frame
        first_frame = clip.get_frame(0)
        img_size = (first_frame.shape[1] , first_frame.shape[0]) # (W , H)

        # can you use moviepy to read the frames from start_frame to end_frame
        assert end_frame - start_frame + 1 == self.len_temporal , f'len(frames) = {len(frames)} != {self.len_temporal}'
        num_frames = int(clip.fps * clip.duration)
        subclip = clip.subclipped(start_frame / clip.fps , (end_frame+1) / clip.fps)

        frames = list(subclip.iter_frames(fps=clip.fps, dtype='uint8'))

        frames = [Image.fromarray(frame).convert('RGB') for frame in frames]
        
        processed_frames = []
        
        for frame in frames:
            img, sz = self.torch_transform(frame)
            processed_frames.append(img)
            # img_size = sz

        # make frames to be of length len_temporal , first by selecting only first len_temporal frames
        # and then by padding the rest with zeros
        frames = processed_frames[:self.len_temporal]
        
        if len(frames) < self.len_temporal:
            for i in range(self.len_temporal - len(frames)):
                if not flip:
                    frames.insert(0 , torch.zeros_like(frames[0]))
                else:
                    frames.append(torch.zeros_like(frames[0]))

        clip = torch.FloatTensor(torch.stack(frames, dim=0)) # (len_temporal , 3 , H , W)
        # clip = clip.permute((0,2,1,3,4)) # 
        clip = clip.permute((1,0,2,3)) # (3 , len_temporal , H , W)
        if flip:
            clip = torch.flip(clip , [1])

        return clip , idx , img_size # return the image size of the first frame
    
    def torch_transform(self , img):
        img_transform = transforms.Compose([
                transforms.Resize((224, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
        ])
        sz = img.size
        img = img_transform(img)
        return img, sz

# %%
saliency_prediction_dataset = SaliencyPredictionDataset(VIDEO_DIR , OUTPUT_DIR)
saliency_prediction_dataloader = DataLoader(saliency_prediction_dataset , batch_size = BATCH_SIZE , shuffle = False)

# %%
model = VideoSaliencyModel(
    transformer_in_channel=32, 
    nhead=4,
    use_upsample=True,
    num_hier=3,
    num_clips=32   
)
model.load_state_dict(torch.load(FILE_WEIGHT))
model.to(device)
model.eval()

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)


def post_process(smap, save_path, img_size):        
    smap = smap.numpy()
    smap = cv2.resize(smap, (img_size[0], img_size[1]))
    smap = blur(smap)
    
    img_save(smap, save_path, normalize=True)


# process the dataloader
for i, (clip, idx, img_size) in tqdm.tqdm(enumerate(saliency_prediction_dataloader), total=len(saliency_prediction_dataloader)):
    with torch.no_grad():
        output = model(clip.to(device))
    num_samples = clip.shape[0]
    for j in range(num_samples):
        smap = output[j].cpu().data
        img_size_ = (img_size[0][j].item() , img_size[1][j].item())
        post_process(smap, saliency_prediction_dataset.samples[idx[j]][3], img_size_)
