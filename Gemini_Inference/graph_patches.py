import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def hh_mm_ss_to_seconds(timestamps):
    hh, mm, ss = timestamps.split(':')
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def get_duration(episode_path):
    # usee ffmpeg to get duration of episode given its path
   
    
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        episode_path
    ]
    duration = float(subprocess.check_output(cmd))
    return duration


def create_graph(episode_folder, episode_id, filtered_timestamps):
    if not filtered_timestamps:
        print("No timestamps to plot.")
        return

    segments = []
    for item in filtered_timestamps:
        # print(item)
        timestamp = item['timestamp']
        start_seg, end_seg = timestamp.split('-')
        start = hh_mm_ss_to_seconds(start_seg)
        end = hh_mm_ss_to_seconds(end_seg)
        category = item['category']
        segments.append({
            "start": start,
            "end": end,
            "category": category
        })

    episode_duration = get_duration(f'{episode_folder}/{episode_id}.mp4')
    if episode_duration == 0:
        print("Episode duration is 0. Skipping plot.")
        return

    # Map categories to vertical position
    categories = sorted(set(seg["category"] for seg in segments))
    category_to_y = {cat: i for i, cat in enumerate(categories)}

    # Plot setup
    fig_height = max(len(categories) * 1, 5)  # ensure non-zero figure height
    fig, ax = plt.subplots(figsize=(12, fig_height))

    colors = plt.cm.get_cmap('tab10', len(categories))

    for seg in segments:
        y = category_to_y[seg["category"]]
        ax.broken_barh([(seg["start"], seg["end"] - seg["start"])], (y - 0.4, 0.8),
                       facecolors=colors(y))

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlim(0, episode_duration)
    ax.set_xlabel("Time (sec)")
    ax.set_title("Shorts Timeline")

    plt.tight_layout()
    plt.savefig(f"{episode_folder}/{episode_id}_category_timeline.png")
    plt.close()
