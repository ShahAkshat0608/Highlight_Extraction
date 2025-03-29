import os
import cv2
import torch
from model import VideoSaliencyModel
import argparse
from utils import *
from os.path import join
from torchvision import transforms
import yaml

import concurrent.futures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


import concurrent.futures
import os
from os.path import join
import torch

def process_directory(dname, path_indata, args, model, len_temporal):
    print(f'processing {dname}', flush=True)
    list_frames = [f for f in os.listdir(os.path.join(path_indata, dname)) if os.path.isfile(os.path.join(path_indata, dname, f))]
    list_frames.sort()
    os.makedirs(join(args.save_path, dname), exist_ok=True)

    # process in a sliding window fashion
    if len(list_frames) >= 2*len_temporal-1:
        snippet = []
        for i in range(len(list_frames)):
            torch_img, img_size = torch_transform(os.path.join(path_indata, dname, list_frames[i]))
            snippet.append(torch_img)
            
            if i >= len_temporal-1:
                clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                clip = clip.permute((0,2,1,3,4))

                process(model, clip, path_indata, dname, list_frames[i], args, img_size)

                # process first (len_temporal-1) frames
                if i < 2*len_temporal-2:
                    process(model, torch.flip(clip, [2]), path_indata, dname, list_frames[i-len_temporal+1], args, img_size)

                del snippet[0]
    else:
        print(f' more frames are needed for {dname}')
    
    return f"Finished processing directory {dname}"



def validate(args):
    path_indata = args.path_indata
    file_weight = args.file_weight
    len_temporal = args.clip_size

    model = VideoSaliencyModel(
        transformer_in_channel=args.transformer_in_channel, 
        nhead=args.nhead,
        use_upsample=bool(args.decoder_upsample),
        num_hier=args.num_hier,
        num_clips=args.clip_size   
    )

    model.load_state_dict(torch.load(file_weight))
    print("Loaded models")

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()
    print(list_indata)
    if args.start_idx != -1:
        _len = (1.0/float(args.num_parts))*len(list_indata)
        list_indata = list_indata[int((args.start_idx-1)*_len): int(args.start_idx*_len)]

    # Process directories in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=config['workers_ViNet']) as executor:
        futures = [
            executor.submit(process_directory, dname, path_indata, args, model, len_temporal) 
            for dname in list_indata
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"Completed: {result}")
            except Exception as e:
                print(f"Directory processing error: {e}")




# def validate(args):
#     path_indata = args.path_indata
#     file_weight = args.file_weight

#     len_temporal = args.clip_size

#     model = VideoSaliencyModel(
#         transformer_in_channel=args.transformer_in_channel, 
#         nhead=args.nhead,
#         use_upsample=bool(args.decoder_upsample),
#         num_hier=args.num_hier,
#      	num_clips=args.clip_size   
#     )

#     model.load_state_dict(torch.load(file_weight))
#     print("Loaded models")

#     model = model.to(device)
#     torch.backends.cudnn.benchmark = False
#     model.eval()

#     list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
#     list_indata.sort()
#     print(list_indata)
#     if args.start_idx!=-1:
#         _len = (1.0/float(args.num_parts))*len(list_indata)
#         list_indata = list_indata[int((args.start_idx-1)*_len): int(args.start_idx*_len)]

#     for dname in list_indata:
#         print ('processing ' + dname, flush=True)
#         list_frames = [f for f in os.listdir(os.path.join(path_indata, dname)) if os.path.isfile(os.path.join(path_indata, dname, f))]
#         list_frames.sort()
#         os.makedirs(join(args.save_path, dname), exist_ok=True)

#         # process in a sliding window fashion
#         if len(list_frames) >= 2*len_temporal-1:

#             snippet = []
#             for i in range(len(list_frames)):
#                 torch_img, img_size = torch_transform(os.path.join(path_indata, dname, list_frames[i]))

#                 snippet.append(torch_img)
                
#                 if i >= len_temporal-1:
#                     clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
#                     clip = clip.permute((0,2,1,3,4))

#                     process(model, clip, path_indata, dname, list_frames[i], args, img_size)

#                     # process first (len_temporal-1) frames
#                     if i < 2*len_temporal-2:
#                         process(model, torch.flip(clip, [2]), path_indata, dname, list_frames[i-len_temporal+1], args, img_size)

#                     del snippet[0]
#         else:
#             print (' more frames are needed')

def torch_transform(path):
    img_transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
    ])
    img = Image.open(path).convert('RGB')
    sz = img.size
    img = img_transform(img)
    return img, sz

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def process(model, clip, path_inpdata, dname, frame_no, args, img_size):
    with torch.no_grad():
        smap = model(clip.to(device)).cpu().data[0]
        
    smap = smap.numpy()
    smap = cv2.resize(smap, (img_size[0], img_size[1]))
    smap = blur(smap)
    
    img_save(smap, join(args.save_path, dname, frame_no), normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_weight',default="./saved_models/ViNet_DHF1K.pt", type=str)
    parser.add_argument('--nhead',default=4, type=int)
    parser.add_argument('--num_encoder_layers',default=3, type=int)
    parser.add_argument('--transformer_in_channel',default=32, type=int)
    parser.add_argument('--save_path',default='/ssd_scratch/cvit/samyak/Results/theatre_hollywood', type=str)
    parser.add_argument('--start_idx',default=-1, type=int)
    parser.add_argument('--num_parts',default=4, type=int)
    parser.add_argument('--path_indata',default='/ssd_scratch/cvit/samyak/DHF1K/val', type=str)
    parser.add_argument('--multi_frame',default=0, type=int)
    parser.add_argument('--decoder_upsample',default=1, type=int)
    parser.add_argument('--num_decoder_layers',default=-1, type=int)
    parser.add_argument('--num_hier',default=3, type=int)
    parser.add_argument('--clip_size',default=32, type=int)
    # open config.yaml
    with open("./config_temp.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])
    args = parser.parse_args()
    
    # change the args path_indata to config['root_folder']
    args.path_indata = f"{config['root_folder']}/{config['episode_id']}/frames"
    args.save_path = f"{config['root_folder']}/{config['episode_id']}/saliency_results"
    args.file_weight = config['vinet_model_path']
    

    # print(args)
    validate(args)

