import os
import torch
from moviepy.editor import VideoFileClip
from music_to_emo.src.emotion_regressor import EmotionRegressor

def get_audio_from_video(input_dir, audio_output_dir):
    # Ensure output directories exist
    os.makedirs(audio_output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            # Construct the full file path
            full_path = os.path.join(input_dir, filename)

            # Load the video file
            try:
                clip = VideoFileClip(full_path)
            except Exception as e:
                print(f"Error loading video {filename}: {e}")
                continue

            # Extract audio from the video and save as MP3
            audio_filename = os.path.splitext(filename)[0] + '.mp3'
            audio_path = os.path.join(audio_output_dir, audio_filename)
            if clip.audio:
                clip.audio.write_audiofile(audio_path)
            else:
                print(f"No audio found in {filename}")
            # Close the clip to free resources
            clip.close()

def convert_to_30fps(input_dir, output_dir):
     # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the directory
    for filename in os.listdir(input_dir):
        # Check for video files (assuming .mp4, .avi, .mov files)
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join("output_dir, filename")
            
            # Convert the video to 30 fps
            try:
                clip = VideoFileClip(input_path)
                clip.set_fps(30).write_videofile(output_path, codec='libx264', audio_codec='aac')
                clip.close()
                print(f"Processed {filename} successfully.")
            except Exception as e:
                print(f"Failed to process {filename}. Error: {e}")



def assign_emotion_to_video(video_dir, audio_dir, output_path):
    # Load regession model
    checkpoint = '../../music-to_emo/saved_models/music_emo_reg.ckpt'
    emo_reg = EmotionRegressor.load_from_checkpoint(checkpoint)
    emo_reg.eval()
    emo_reg.freeze()

    #load aaudio feature extractor
    vggish = torch.hub.load('harritaylor/torchvggish', 'vggish').eval()

    #ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #iterate over all files in the directory
    for filename in os.listdir(video_dir):
        #load the video
        if filename.endswith('.mp4'):
            full_path = os.path.join(video_dir, filename)
            clip = VideoFileClip(full_path)

        #get the name of the file without the extension
        track_id = os.path.basename(filename)[:-4]
        audio_path = os.path.join(audio_dir, f"{track_id}.mp3")

        #get the featurres frm the audio
        features = vggish(audio_path).detach()
        features = features.unsqueeze(0)

        #get the predictions
        preds = emo_reg(features)




    
