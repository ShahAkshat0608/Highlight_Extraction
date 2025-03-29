import yaml
import os
import subprocess

# import the config.yaml file
def get_frames(config):
    # run a bash command
    os.system(f'bash extract_frames.sh {config["root_folder"]}/{config["episode_id"]}')