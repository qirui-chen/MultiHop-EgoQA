import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge('torch')

from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os
import argparse
import json


def process_videos(video_dir, partition, total_partitions):

    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]

    num_videos = len(video_files)
    videos_per_partition = (num_videos + total_partitions - 1) // total_partitions  
    start_idx = partition * videos_per_partition
    end_idx = min((partition + 1) * videos_per_partition, num_videos)
    
    model_id = '<PATH>/llava-v1.6-mistral-7b-hf/'
    processor = LlavaNextProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)

    results = {}

    for video_file in tqdm(video_files[start_idx:end_idx]):
        # prepare video reader
        vr = VideoReader(video_file, ctx=cpu(0))
        total_frames = len(vr)

        # calculate frame interval to get 90 frames
        NUM_FRAMES = 180
        FPS = 30
        frame_interval = max(1, total_frames // NUM_FRAMES)
        frames = [i * frame_interval for i in range(NUM_FRAMES)]

        captions = []
        batch_size = 32
        prompt_template = "[INST] <image>\nDescribe the action in this video frame in one concise sentence. [/INST]"

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            images = vr.get_batch(batch_frames).permute(0, 3, 1, 2)  # assuming the images are in NHWC format
            prompts = [prompt_template] * len(images)
            inputs = processor(prompts, images, return_tensors='pt', padding=True).to("cuda")
            processor.tokenizer.padding_side = "left"

            # generate captions
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)

            for idx, output in enumerate(outputs):
                caption = processor.decode(output, skip_special_tokens=True).split('[/INST] ')[-1]
                frame_time = int(batch_frames[idx] / FPS)  # calculate the time in the video
                captions.append([frame_time, caption.strip()])

        video_id = os.path.basename(video_file).split('.')[0]
        results[video_id] = captions


    with open(f'captions_part_{partition}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Captions for partition {partition} saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and generate captions.")
    parser.add_argument('--partition', type=int, required=True, help="Specify which partition to process.")
    parser.add_argument('--total_partitions', type=int, required=True, help="Specify the total number of partitions.")
    args = parser.parse_args()

    video_dir = '<Video Path>'
    process_videos(video_dir, args.partition, args.total_partitions)
