import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm

import numpy as np
class DiceMetricsWrapper:
    """
    Wrapper class for computing Dice metric and HD95 during training and validation.
    """
    def __init__(self, num_classes=2, is_2d=False, compute_hd95=True):
        """
        Initialize metric wrapper.
        
        Args:
            num_classes: Number of output classes
            include_background: Whether to include background class in metric
            compute_hd95: Whether to compute Hausdorff Distance 95
        """
        self.num_classes = num_classes
        self.compute_hd95 = compute_hd95
        self.is_2d = is_2d
        
        # Dice metric
        self.dice_metric = DiceMetric(
            reduction="mean", 
            get_not_nans=False
        )
        
        # HD95 metric
        if compute_hd95:
            self.hd95_metric = HausdorffDistanceMetric(
                include_background=True,
                percentile=95,
                reduction="mean",
                get_not_nans=False
            )
        
        
        # Transforms for predictions and labels
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes, dim=1)
        self.post_label = AsDiscrete(to_onehot=num_classes, dim=1)
        
        # Accumulators for additional metrics
        self.total_correct = 0
        self.total_pixels = 0
        self.intersection_sum = 0
        self.union_sum = 0
    
    def update(self, output, target):
        """
        Update metrics with a batch of predictions.
        
        Args:
            output: Model output logits (B, C, H, W, D)
            target: Ground truth labels (B, 1, H, W, D) or (B, H, W, D)
        """
        # Apply softmax to get probabilities
        output_probs = F.softmax(output, dim=1)
        
        # Convert to one-hot format for MONAI metrics
        pred_onehot = self.post_pred(output_probs)
        if self.is_2d:
            target = target.unsqueeze(1)  # Add channel dim if missing
        target_onehot = self.post_label(target)
        
        # Update Dice metric
        self.dice_metric(y_pred=pred_onehot, y=target_onehot)
        
        # Update HD95 if enabled
        if self.compute_hd95:
            self.hd95_metric(y_pred=pred_onehot, y=target_onehot)
        
        # Compute pixel accuracy and IoU
        pred_labels = torch.argmax(output, dim=1)
        target_labels = target.squeeze(1) if target.dim() == 5 else target
        
        # Pixel accuracy
        correct = (pred_labels == target_labels).sum().item()
        total = target_labels.numel()
        self.total_correct += correct
        self.total_pixels += total
        
        # IoU computation
        for class_idx in range(self.num_classes):
            pred_mask = (pred_labels == class_idx)
            target_mask = (target_labels == class_idx)
            
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            
            self.intersection_sum += intersection
            self.union_sum += union
    
    def compute_all(self):
        """
        Compute all metrics and return as dictionary.
        
        Returns:
            Dictionary with mean_dice, mean_iou, pixel_accuracy, and hd95 (if enabled)
        """
        # Aggregate Dice metric
        dice_score = self.dice_metric.aggregate().item()
        
        # Compute pixel accuracy
        pixel_accuracy = self.total_correct / self.total_pixels if self.total_pixels > 0 else 0.0
        
        # Compute mean IoU
        mean_iou = self.intersection_sum / self.union_sum if self.union_sum > 0 else 0.0
        
        results = {
            "mean_dice": dice_score,
            "mean_iou": mean_iou,
            "pixel_accuracy": pixel_accuracy
            
        }
        
        # Add HD95 if enabled
        if self.compute_hd95:
            hd95_score = self.hd95_metric.aggregate().item()
            results["hd95"] = hd95_score
        
        return results
    
    def __call__(self, output, target):
        """
        Compute Dice metric and HD95 for a single batch.
        
        Args:
            output: Model output logits (B, C, H, W, D)
            target: Ground truth labels (B, 1, H, W, D)
            
        Returns:
            Dictionary with dice_score and hd95 (if enabled)
        """
        # Apply softmax to get probabilities
        output_probs = F.softmax(output, dim=1)
        
        # Convert to one-hot format
        pred_onehot = self.post_pred(output_probs)
        if self.is_2d:
            target = target.unsqueeze(1)  # Add channel dim if missing
        target_onehot = self.post_label(target)
        
        # Compute Dice metric
        self.dice_metric(y_pred=pred_onehot, y=target_onehot)
        dice_score = self.dice_metric.aggregate().item()
        
        results = {"dice_score": dice_score}
        
        # Compute HD95 if enabled
        if self.compute_hd95:
            self.hd95_metric(y_pred=pred_onehot, y=target_onehot)
            hd95_score = self.hd95_metric.aggregate().item()
            results["hd95"] = hd95_score
        
        return results
    
    def reset(self):
        """Reset the metric state."""
        self.dice_metric.reset()
        if self.compute_hd95:
            self.hd95_metric.reset()
        
        # Reset accumulators
        self.total_correct = 0
        self.total_pixels = 0
        self.intersection_sum = 0
        self.union_sum = 0
    
    def get_buffer(self):
        """Get current metric buffer."""
        buffer = {"dice": self.dice_metric.get_buffer()}
        if self.compute_hd95:
            buffer["hd95"] = self.hd95_metric.get_buffer()
        return buffer
    
def save_checkpoint(model, optimizer, epoch, log_dir, model_name):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_path = log_dir / f"{model_name}_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")



#Reference
#https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unet_segmentation_3d_ignite.ipynb