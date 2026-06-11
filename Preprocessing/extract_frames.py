import argparse
import os
from pathlib import Path

import cv2


def extract_frames(video_path, frames_dir):
    video_name = Path(video_path).stem

    video_frames_dir = os.path.join(frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(video_frames_dir, f"{frame_count}.jpg")

        cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {video_frames_dir}")
    return video_name, frame_count


def main(videos_dir, frames_dir, annotations_file):
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
    videos_dir = []
    with open("../../shared-nvme/UCF-crime/Anomaly_Train.txt", "r", encoding="utf-8") as f:
    
    
        for line in f:
            parts = line.strip().split()
            path = f"../../shared-nvme/UCF-crime/{parts[0]}"
            videos_dir.append(path)
            # if len(parts) >= 2:
            #     filename = parts[0]
            #     category = parts[1]
            #     if category!="Normal":
            #         path = f"../../shared-nvme/UCF-crime/anomaly/{category}/{filename}"
            #         videos_dir.append(path)

    with open(annotations_file, "w") as f:

        for video_file in videos_dir:
            if video_file.endswith(".avi") or video_file.endswith(".mp4"):
                video_path = video_file
                id = video_file.split("Videos")[-1].split("_")[0]
                if 'Normal' in video_path and int(id) >= 369:
                    if os.path.exists(video_path):
                        video_name, num_frames = extract_frames(video_path, frames_dir)
                        f.write(f"{video_name} 0 {num_frames - 1} 0\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=False,
        help="Directory path to the videos.",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=False,
        help="Directory path to the frames.",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        required=False,
        help="Path to the annotations file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.videos_dir = "../../shared-nvme/UCF-crime/Testing_Normal_Videos_Anomaly/"
    args.frames_dir = "../../shared-nvme/UCF-crime/frames_train"
    args.annotations_file = "../../shared-nvme/UCF-crime/train_normal.txt"
    main(args.videos_dir, args.frames_dir, args.annotations_file)
