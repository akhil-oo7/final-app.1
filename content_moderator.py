from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels, feature_extractor):
        self.frames = frames
        self.labels = labels
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Preprocess the image
        inputs = self.feature_extractor(image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ContentModerator:
    def __init__(self, model_name="microsoft/resnet-50", train_mode=False):
        """
        Initialize the ContentModerator with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            train_mode (bool): Whether to initialize in training mode
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Always use feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        if train_mode:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification: violent vs non-violent
                ignore_mismatched_sizes=True
            ).to(self.device)
        else:
            # Load our trained model
            model_path = os.path.join("models", "best_model")
            if os.path.exists(model_path):
                print("Loading trained model...")
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_path,
                    num_labels=2
                ).to(self.device)
                self.model.eval()  # Set to evaluation mode
            else:
                raise FileNotFoundError("Trained model not found. Please train the model first.")
    
    def analyze_frames(self, frames):
        """
        Analyze frames for inappropriate content.
        
        Args:
            frames (list): List of video frames as numpy arrays
            
        Returns:
            list: List of analysis results for each frame
        """
        results = []
        
        # Convert frames to dataset
        dataset = VideoFrameDataset(frames, [0] * len(frames), self.feature_extractor)
        dataloader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                outputs = self.model(pixel_values)
                predictions = torch.softmax(outputs.logits, dim=1)
                
                for pred in predictions:
                    # Get probability of violence (class 1)
                    violence_prob = pred[1].item()
                    # Lower threshold for violence detection
                    flagged = violence_prob > 0.3  # Changed from 0.5 to 0.3
                    
                    results.append({
                        'flagged': flagged,
                        'reason': "Detected violence" if flagged else "No inappropriate content detected",
                        'confidence': violence_prob if flagged else 1 - violence_prob
                    })
        
        return results

    def analyze_video(video_path):
        print("✅ Starting video analysis...")
        try:
            # Frame extraction with validation
            frames = extract_frames(video_path)
            if not frames or not isinstance(frames, list):
                print("❌ Frame extraction failed - empty result or invalid type")
                return {"status": "error", "message": "Frame extraction failed"}
                
            print(f"✅ {len(frames)} frames extracted")
            if len(frames[0].shape) != 3:  # Check frame dimensions
                print(f"❌ Invalid frame shape: {frames[0].shape}")
                return {"status": "error", "message": "Invalid frame dimensions"}
    
            # Model prediction
            try:
                predictions = classify_frames(frames)
                print(f"✅ Predictions sample: {predictions[:2]}")
            except Exception as e:
                print(f"❌ Prediction failed: {str(e)}")
                return {"status": "error", "message": "Model prediction failed"}
    
            # Result processing
            try:
                summary = summarize_results(predictions)
                print(f"✅ Analysis completed successfully")
                return {"status": "success", "results": summary}
            except Exception as e:
                print(f"❌ Result processing failed: {str(e)}")
                return {"status": "error", "message": "Result processing failed"}
                
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return {"status": "error", "message": "Unexpected analysis error"}

    def load_model():
        try:
            if not hasattr(self, 'model'):
                print("⚙️ Loading model...")
                self.model = torch.load(MODEL_PATH, map_location=self.device)
                self.model.eval()
                print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            raise