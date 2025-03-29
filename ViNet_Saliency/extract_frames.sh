root=$1
mkdir -p "$root/frames"  # Create the frames directory

# Loop through each video file in the raw_videos directory
for video in "$root/raw_videos"/*; do
    # Extract the filename from the full path
    filename=$(basename "$video")
    
    # Create a folder for the frames using the filename (without the extension)
    foldername="$root/frames/${filename%.*}"

    mkdir -p "$foldername"  # Create the folder for frames

    # Run ffmpeg to extract frames from the video
    ffmpeg -i "$video" "$foldername/frame_%04d.png"
done

