"""
Inference script for multi-task crop and growth-stage classification
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json
import config
from model import create_model

class CropStagePredictor:
    """Predictor for crop and growth-stage classification"""
    
    def __init__(self, model_path, config_path, class_names_path, device=None):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training configuration
            class_names_path: Path to class names JSON
            device: Device to run inference on (default: from config)
        """
        self.device = device if device else config.DEVICE
        
        # Load class names
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        self.crop_classes = class_names['crop_classes']
        self.stage_classes = class_names['stage_classes']
        
        # Load training configuration
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        
        self.image_size = training_config['image_size']
        self.mean = config.MEAN
        self.std = config.STD
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Load model
        self.model = create_model(config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        print(f"Crop classes: {len(self.crop_classes)}")
        print(f"Stage classes: {len(self.stage_classes)}")
    
    def predict(self, image_path):
        """
        Predict crop and growth-stage for an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with predictions and confidences
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            crop_logits, stage_logits = self.model(image_tensor)
            
            # Get probabilities
            crop_probs = F.softmax(crop_logits, dim=1)
            stage_probs = F.softmax(stage_logits, dim=1)
            
            # Get predictions
            crop_pred_idx = torch.argmax(crop_probs, dim=1).item()
            stage_pred_idx = torch.argmax(stage_probs, dim=1).item()
            
            # Get confidences
            crop_confidence = crop_probs[0, crop_pred_idx].item()
            stage_confidence = stage_probs[0, stage_pred_idx].item()
            
            # Get class names
            crop_pred = self.crop_classes[crop_pred_idx]
            stage_pred = self.stage_classes[stage_pred_idx]
            
            # Get all class probabilities
            crop_class_probs = {
                self.crop_classes[i]: crop_probs[0, i].item()
                for i in range(len(self.crop_classes))
            }
            
            stage_class_probs = {
                self.stage_classes[i]: stage_probs[0, i].item()
                for i in range(len(self.stage_classes))
            }
        
        return {
            "crop": crop_pred,
            "crop_confidence": crop_confidence,
            "crop_probabilities": crop_class_probs,
            "growth_stage": stage_pred,
            "stage_confidence": stage_confidence,
            "stage_probabilities": stage_class_probs
        }
    
    def predict_batch(self, image_paths):
        """
        Predict for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = str(image_path)
                result['success'] = True
            except Exception as e:
                result = {
                    'image_path': str(image_path),
                    'success': False,
                    'error': str(e)
                }
            results.append(result)
        
        return results

def main():
    """Main function for standalone inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crop and growth-stage inference')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Path to output JSON file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CropStagePredictor(
        model_path=config.CHECKPOINT_DIR / "best_model.pth",
        config_path=config.ARTIFACTS_DIR / "training_config.json",
        class_names_path=config.ARTIFACTS_DIR / "class_names.json"
    )
    
    # Run inference
    if args.image:
        # Single image
        result = predictor.predict(args.image)
        print(json.dumps(result, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    
    elif args.image_dir:
        # Batch inference
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.png'))
        
        print(f"Found {len(image_paths)} images")
        
        results = predictor.predict_batch(image_paths)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"Successfully processed {successful}/{len(results)} images")
    
    else:
        print("Please provide --image or --image_dir")

if __name__ == "__main__":
    main()