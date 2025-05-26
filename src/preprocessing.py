import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
from mtcnn import MTCNN
from tqdm import tqdm
import argparse
import uuid

# Checkpoint system configuration
CHECKPOINT_FILE = "processed_videos.log"
BATCH_SIZE = 500  # Number of videos processed per batch

def extract_faces_from_video(video_path, output_dir, frames_per_video=30):
    """Extract faces from video frames using MTCNN face detection."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return 0

        detector = MTCNN()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        frame_indices = []
        if total_frames > 0:
            frame_indices = sorted(list(
                {int(total_frames * (i / frames_per_video)) 
                for i in range(frames_per_video)}
            ))
        else:
            print(f"Warning: {video_path} has 0 frames. Skipping.")
            return 0

        saved_count = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect faces
            faces = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if faces:
                # Get largest face
                main_face = max(faces, key=lambda x: x['confidence'])
                x, y, w, h = main_face['box']
                
                # Ensure coordinates are within frame boundaries
                x, y = max(0, x), max(0, y)
                face = frame[y:y+h, x:x+w]
                
                if face.size == 0:
                    continue

                # Save face
                resized = cv2.resize(face, (224, 224))
                img_name = f"{uuid.uuid4()}.jpg"
                output_path = os.path.join(output_dir, img_name)
                
                if cv2.imwrite(output_path, resized):
                    saved_count += 1
                else:
                    print(f"Failed to save image: {output_path}")

        cap.release()
        return saved_count

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return 0

def load_processed_videos():
    """Load set of already processed videos from checkpoint file."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, 'r') as f:
        return set(line.strip() for line in f)

def save_processed_video(video_name):
    """Append processed video to checkpoint file."""
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(f"{video_name}\n")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, 
                       help='Directory containing input videos')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save extracted face images')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                       help=f'Videos per batch (default: {BATCH_SIZE})')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--restart', action='store_true',
                       help='Delete checkpoints and restart processing')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Print critical paths
    print(f"\n{'='*40}")
    print("Deepfake Video Preprocessing Pipeline")
    print(f"{'='*40}")
    print(f"Absolute input path: {os.path.abspath(args.input_dir)}")
    print(f"Absolute output path: {os.path.abspath(args.output_dir)}")
    print(f"Checkpoint file: {os.path.abspath(CHECKPOINT_FILE)}")
    print(f"{'='*40}\n")

    # Handle restart flag
    if args.restart:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("Checkpoint file deleted. Starting fresh processing.")

    # Load processed videos
    processed_videos = load_processed_videos()
    
    # Get sorted list of video files
    video_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ])
    
    # Determine unprocessed videos
    unprocessed = [f for f in video_files if f not in processed_videos]
    
    # Print status
    if args.resume:
        print(f"Resuming processing: {len(unprocessed)}/{len(video_files)} videos remaining")
    else:
        print(f"Total videos found: {len(video_files)}")

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process in batches
    total_processed = 0
    for batch_idx in range(0, len(unprocessed), args.batch):
        batch_files = unprocessed[batch_idx:batch_idx + args.batch]
        
        print(f"\n{'='*40}")
        print(f"Processing batch {(batch_idx//args.batch)+1}/{(len(unprocessed)-1)//args.batch + 1}")
        print(f"Videos {batch_idx+1}-{min(batch_idx+args.batch, len(unprocessed))} of {len(unprocessed)}")
        print(f"{'='*40}")

        for video_file in tqdm(batch_files, desc="Videos"):
            video_path = os.path.join(args.input_dir, video_file)
            try:
                faces_saved = extract_faces_from_video(video_path, args.output_dir)
                
                if faces_saved > 0:
                    save_processed_video(video_file)
                    total_processed += 1
                    print(f" ✔ {video_file}: {faces_saved} faces saved")
                else:
                    print(f" ✖ {video_file}: No faces detected")

            except Exception as e:
                print(f"\nCritical error processing {video_file}: {str(e)}")
                continue

    # Final report
    print(f"\n{'='*40}")
    print(f"Processing complete!")
    print(f"Total videos processed: {total_processed}")
    print(f"Face images saved to: {os.path.abspath(args.output_dir)}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
