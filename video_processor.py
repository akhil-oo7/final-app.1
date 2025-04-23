import cv2
import numpy as np
from tqdm import tqdm
import gc

class VideoProcessor:
    def __init__(self, frame_interval=30, target_size=(224, 224)):
        """
        Initialize the VideoProcessor.
        
        Args:
            frame_interval (int): Number of frames to skip between extractions
            target_size (tuple): Target size for frame resizing (height, width)
        """
        self.frame_interval = frame_interval
        self.target_size = target_size
    
    def extract_frames(self, video_path, max_frames=None):
        """
        Extract frames from a video file with memory optimizations.
        
        Args:
            video_path (str): Path to the video file
            max_frames (int): Maximum number of frames to extract (optional)
            
        Returns:
            list: List of extracted frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames * self.frame_interval)
        
        try:
            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                frame_count = 0
                while frame_count < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % self.frame_interval == 0:
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_resized = cv2.resize(frame_rgb, self.target_size)
                            frames.append(frame_resized)
                            
                            # Periodically clear memory
                            if len(frames) % 10 == 0:
                                gc.collect()
                                
                            # Early exit if we reached max_frames
                            if max_frames and len(frames) >= max_frames:
                                break
                                
                        except Exception as e:
                            print(f"Error processing frame {frame_count}: {str(e)}")
                            continue
                            
                    frame_count += 1
                    pbar.update(1)
                    
        finally:
            cap.release()
            
        return frames