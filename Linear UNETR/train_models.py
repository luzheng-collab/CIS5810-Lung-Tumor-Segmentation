import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import numpy as np
import tqdm
import time
import pandas as pd

from models import load_model, save_model
from data_loader import load_image_data,save_checkpoint
from metric import DiceMetricsWrapper
from thop import profile, clever_format


# === MONAI Imports ===
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.data import decollate_batch
from monai.utils.enums import MetricReduction



class TrainMedicalSeg:
    def __init__(
        self,
        exp_dir: str = "logs",
        model_name: str = "UNet2D",
        num_epoch: int = 50,
        lr: float = 0.001,
        batch_size: int = 128,
        seed: int = 2025,
        in_channels: int = 1,
        num_classes: int = 2,
        train_data_dir: str = "../data/train",
        val_data_dir: str = "../data/val",
        test_data_dir: str = "../data/test",
        accumulation_steps: int = 6,
        kfold_idx: int = 0,
        use_amp: bool = True,
        **kwargs,
    ):
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.num_epoch = num_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.train_data_dir = f"../data_new/train/fold{kfold_idx}/train"
        self.val_data_dir = f"../data_new/train/fold{kfold_idx}/val"
        self.test_data_dir = f"../data_new/test"
        self.accumulation_steps = accumulation_steps
        self.kwargs = kwargs
        self.kfold_idx = kfold_idx
        self.use_amp = use_amp

        # Performance monitoring attributes
        self.batch_times = []
        self.peak_memory_bytes = []
        self.forward_flops = None
        self.estimated_flops_per_batch = None
        self.warmup_batches = 0
        self.metrics_history = []
        self.best_val_dice_fg = 0.0
      
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA available")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            print("Arm Macs")
            self.use_amp = False  # MPS doesn't support AMP
        else:
            print("CUDA not available, using CPU")
            self.device = torch.device("cpu")
            self.use_amp = False  # CPU doesn't support AMP
        
        if self.use_amp:
            print(f"Mixed Precision Training: ENABLED")
        else:
            print(f"Mixed Precision Training: DISABLED")
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model = None
        self.optimizer = None

    def _calculate_model_flops(self):
        """Calculate FLOPs for the model using a dummy input."""
        if profile is None:
            print("Warning: thop not installed. Skipping FLOPs calculation.")
            return
        
        try:
            # Create dummy input based on model type
            if self.model_name == "UNet2D":
                dummy_input = torch.randn(1, self.in_channels, 256, 256).to(self.device)
            else:
                dummy_input = torch.randn(1, self.in_channels, 64, 64, 64).to(self.device)
            
            macs, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            self.forward_flops = macs * 2  # FLOPs = 2 * MACs
            self.estimated_flops_per_batch = self.forward_flops * self.batch_size * 3  # 3x for training
            
            print(f"Model FLOPs (single forward): {self.forward_flops / 1e9:.2f} GFLOPs")
            print(f"Estimated FLOPs per batch: {self.estimated_flops_per_batch / 1e9:.2f} GFLOPs")
        except Exception as e:
            print(f"Error calculating FLOPs: {e}")

    def _get_avg_batch_latency_sec(self):
        """Calculate average batch latency excluding warmup batches."""
        if len(self.batch_times) > self.warmup_batches:
            return sum(self.batch_times[self.warmup_batches:]) / (len(self.batch_times) - self.warmup_batches)
        elif self.batch_times:
            return sum(self.batch_times) / len(self.batch_times)
        return None

    def _save_performance_metrics(self, log_dir):
        """Save performance metrics to CSV file."""
        if self.metrics_history:
            output_path = log_dir / "performance_metrics.csv"
            df_metrics = pd.DataFrame(self.metrics_history)
            df_metrics.to_csv(output_path, index=False)
            print(f"Performance metrics saved to: {output_path}")

    def train_model(self):
        log_dir = Path(self.exp_dir) / f"{self.model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
        self.log_dir= log_dir
        logger = tb.SummaryWriter(log_dir)
        device = self.device
        model_name = self.model_name
        
        # 1. Load Model
        self.model = load_model(model_name, in_channels=self.in_channels, num_classes=self.num_classes)
        print(f'Training model: {model_name}')
        print(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        self.model = self.model.to(device)

        # Calculate FLOPs
        self._calculate_model_flops()

        # Reset CUDA memory stats if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # Load data
        if self.model_name == "UNet2D":
            print("Loading 2D slice-based Training Data...")
            train_data = load_image_data(
                dataset_path=self.train_data_dir,
                label_subdir='label_gtvp',
                batch_size=self.batch_size,
                shuffle=True,
                is_patches=False
            )
            val_data = load_image_data(
                dataset_path=self.val_data_dir,
                label_subdir='label_gtvp',
                batch_size=self.batch_size,
                shuffle=False,
                is_patches=False
            )
        else:
            print("Loading Training Data ")
            train_data = load_image_data(
                dataset_path=self.train_data_dir,
                label_subdir='label_gtvp',
                batch_size=self.batch_size,
                shuffle=True,
                is_patches=True,
                transform_pipeline="aug_train", 
                is_label_sampler=True,            
                use_queue=True                    
            )
            
            print("Loading Validation Data...")
            val_data = load_image_data(
                dataset_path=self.val_data_dir,
                label_subdir='label_gtvp',
                batch_size=self.batch_size,
                shuffle=False,
                is_patches=True,
                transform_pipeline="aug_val",
                use_queue=True
            )

        loss_func = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-4/self.lr, total_iters=1000)
        
        # Initialize GradScaler for mixed precision
        scaler = torch.amp.GradScaler(device='cuda', enabled=self.use_amp)


        # Initialize metrics - use include_background=True to get per-class metrics
        dice_metric_val = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
        dice_metric_train = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
        #HD95 Metrics   
        hd95_metric_train = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
        hd95_metric_val = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)

        # Add DiceMetricsWrapper for mean dice and hd95 computation
        if self.model_name == "UNet2D":
            train_metrics_wrapper = DiceMetricsWrapper(is_2d=True)
            val_metrics_wrapper = DiceMetricsWrapper(is_2d=True)
        else:
            train_metrics_wrapper = DiceMetricsWrapper()
            val_metrics_wrapper = DiceMetricsWrapper()
       
        post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        post_label = AsDiscrete(to_onehot=self.num_classes)

        global_step = 0

        # === Training Loop ===
        for epoch in range(self.num_epoch):
            self.model.train()
            epoch_train_loss = 0.0
            total_epoch_flops = 0.0
            self.batch_times = []
            self.peak_memory_bytes = []
            
            # Reset train metrics at start of epoch
            dice_metric_train.reset()
            hd95_metric_train.reset()
            train_metrics_wrapper.reset()

            train_loader_tqdm = tqdm.tqdm(train_data, desc=f"Epoch {epoch+1}/{self.num_epoch} [Train]")

            for batch_idx, batch in enumerate(train_loader_tqdm):
                # Start timing
                batch_start_time = time.time()
                
                # Reset memory stats for this batch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)

                # Use autocast for mixed precision
                with torch.autocast(device_type='cuda', enabled=self.use_amp):
                    if self.model_name == "UNet2D":
                        img, mask = batch['image'].to(device).float(), batch['label'].to(device)
                        mask_unsq = mask.unsqueeze(1).long()
                        logits = self.model(img)
                        loss = loss_func(logits, mask_unsq)
                        current_batch_size = img.shape[0]
                    else:
                        img = batch['image']['data'].to(device).float()
                        mask = batch['label']['data'].to(device)
                        logits = self.model(img)
                        loss = loss_func(logits, mask)
                        current_batch_size = img.shape[0]
                
                # --- Update Training Dice Metrics ---
                with torch.no_grad():
                    train_outputs_list = decollate_batch(logits)
                    if mask.dim() == 4: 
                        mask_for_metric = mask.unsqueeze(1)
                    else:
                        mask_for_metric = mask
                    
                    if self.model_name == "UNet2D":
                         mask_for_metric = mask_unsq

                    train_labels_list = decollate_batch(mask_for_metric)
                    
                    train_output_convert = [post_pred(pred) for pred in train_outputs_list]
                    train_label_convert = [post_label(label) for label in train_labels_list]
                    
                    # update train metrics (MONAI for per-class)
                    dice_metric_train(y_pred=train_output_convert, y=train_label_convert)
                    hd95_metric_train(y_pred=train_output_convert, y=train_label_convert)
                    
                    # update DiceMetricsWrapper for mean values
                    if self.model_name == "UNet2D":
                        train_metrics_wrapper.update(logits, mask)
                    else:
                        train_metrics_wrapper.update(logits, mask)

                
                # Backprop with gradient scaling
                loss_per_step = loss / self.accumulation_steps
                scaler.scale(loss_per_step).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_data):
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    
                    # Step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_train_loss += loss.item() * self.accumulation_steps
                
                # Track batch time
                batch_duration = time.time() - batch_start_time
                self.batch_times.append(batch_duration)
                
                # Track memory usage
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.max_memory_allocated(device)
                    self.peak_memory_bytes.append(mem_allocated)
                
                # Track FLOPs
                if self.forward_flops is not None:
                    batch_flops = self.forward_flops * current_batch_size * 3
                    total_epoch_flops += batch_flops
                
                global_step += 1
                
                
            
            # Aggregate Training Metrics - MONAI for per-class values
            dice_train_per_class = dice_metric_train.aggregate()  # Shape: [num_classes]
            hd95_train_per_class = hd95_metric_train.aggregate()  # Shape: [num_classes]
            
            # Extract background (class 0) and foreground (class 1) metrics
            dice_train_bg = dice_train_per_class[0].item()
            dice_train_fg = dice_train_per_class[1].item()
            
            hd95_train_bg = hd95_train_per_class[0].item()
            hd95_train_fg = hd95_train_per_class[1].item()
            
            # Get mean values from DiceMetricsWrapper
            train_results = train_metrics_wrapper.compute_all()
            dice_train_mean = train_results['mean_dice']
            hd95_train_mean = train_results['hd95']

            # === Validation Loop ===
            with torch.inference_mode():
                self.model.eval()
                epoch_val_loss = 0.0
                dice_metric_val.reset() 
                hd95_metric_val.reset()
                val_metrics_wrapper.reset()

                val_loader_tqdm = tqdm.tqdm(val_data, desc=f"Epoch {epoch+1}/{self.num_epoch} [Val]")

                for batch in val_loader_tqdm:
                    # Use autocast for validation too
                    with torch.autocast(device_type='cuda', enabled=self.use_amp):
                        if self.model_name == "UNet2D":
                            img, mask = batch['image'].to(device).float(), batch['label'].to(device)
                            mask_unsq = mask.unsqueeze(1).long()
                            logits = self.model(img)
                            loss = loss_func(logits, mask_unsq)
                        else:
                            img = batch['image']['data'].to(device).float()
                            mask = batch['label']['data'].to(device)
                            logits = self.model(img)
                            loss = loss_func(logits, mask)
                
                    epoch_val_loss += loss.item()

                    val_outputs_list = decollate_batch(logits)
                    
                    if mask.dim() == 4: 
                        mask = mask.unsqueeze(1)
                    val_labels_list = decollate_batch(mask)

                    # Apply Transforms
                    val_output_convert = [post_pred(val_pred) for val_pred in val_outputs_list]
                    val_label_convert = [post_label(val_label) for val_label in val_labels_list]

                    # Update val Metric (MONAI for per-class)
                    dice_metric_val(y_pred=val_output_convert, y=val_label_convert)
                    hd95_metric_val(y_pred=val_output_convert, y=val_label_convert)
                    
                    # Update DiceMetricsWrapper for mean values
                    if self.model_name == "UNet2D":
                        val_metrics_wrapper.update(logits, mask)
                    else:
                        val_metrics_wrapper.update(logits, mask)

                # Aggregate results - MONAI for per-class values
                dice_val_per_class = dice_metric_val.aggregate()  # Shape: [num_classes]
                hd95_val_per_class = hd95_metric_val.aggregate()  # Shape: [num_classes]
                
                # Extract background (class 0) and foreground (class 1) metrics
                dice_val_bg = dice_val_per_class[0].item()
                dice_val_fg = dice_val_per_class[1].item()
                
                hd95_val_bg = hd95_val_per_class[0].item()
                hd95_val_fg = hd95_val_per_class[1].item()
                
                # Get mean values from DiceMetricsWrapper
                val_results = val_metrics_wrapper.compute_all()
                dice_val_mean = val_results['mean_dice']
                hd95_val_mean = val_results['hd95']
                
                avg_train_loss = epoch_train_loss / len(train_data)
                avg_val_loss = epoch_val_loss / len(val_data)

                # Calculate performance metrics
                epoch_metrics = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_dice_mean": dice_train_mean,
                    "train_dice_bg": dice_train_bg,
                    "train_dice_fg": dice_train_fg,
                    "train_hd95_mean": hd95_train_mean,
                    "train_hd95_bg": hd95_train_bg,
                    "train_hd95_fg": hd95_train_fg,
                    "val_dice_mean": dice_val_mean,
                    "val_dice_bg": dice_val_bg,
                    "val_dice_fg": dice_val_fg,
                    "val_hd95_mean": hd95_val_mean,
                    "val_hd95_bg": hd95_val_bg,
                    "val_hd95_fg": hd95_val_fg,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                }

                # Add batch latency
                avg_batch_latency = self._get_avg_batch_latency_sec()
                if avg_batch_latency is not None:
                    epoch_metrics["avg_batch_latency_sec"] = avg_batch_latency
                    logger.add_scalar("performance/avg_batch_latency_sec", avg_batch_latency, epoch)

                # Add memory usage
                if self.peak_memory_bytes:
                    max_peak_memory_bytes = max(self.peak_memory_bytes)
                    max_peak_memory_mb = max_peak_memory_bytes / (1024 * 1024)
                    epoch_metrics["max_peak_memory_bytes"] = max_peak_memory_bytes
                    epoch_metrics["max_peak_memory_mb"] = max_peak_memory_mb
                    logger.add_scalar("performance/max_peak_memory_mb", max_peak_memory_mb, epoch)

                # Add FLOPs
                if total_epoch_flops > 0:
                    epoch_metrics["total_training_flops_per_epoch"] = total_epoch_flops
                    logger.add_scalar("performance/total_training_tflops", total_epoch_flops / 1e12, epoch)

                # Store metrics
                self.metrics_history.append(epoch_metrics)

                # Tensorboard Logging - separate by class
                logger.add_scalar("train/dice_mean", dice_train_mean, epoch)
                logger.add_scalar("train/dice_background", dice_train_bg, epoch)
                logger.add_scalar("train/dice_foreground", dice_train_fg, epoch)
                logger.add_scalar("train/hd95_mean", hd95_train_mean, epoch)
                logger.add_scalar("train/hd95_background", hd95_train_bg, epoch)
                logger.add_scalar("train/hd95_foreground", hd95_train_fg, epoch)
                
                logger.add_scalar("val/dice_mean", dice_val_mean, epoch)
                logger.add_scalar("val/dice_background", dice_val_bg, epoch)
                logger.add_scalar("val/dice_foreground", dice_val_fg, epoch)
                logger.add_scalar("val/hd95_mean", hd95_val_mean, epoch)
                logger.add_scalar("val/hd95_background", hd95_val_bg, epoch)
                logger.add_scalar("val/hd95_foreground", hd95_val_fg, epoch)

                logger.add_scalar("train/avg_loss", avg_train_loss, epoch)
                logger.add_scalar("val/avg_loss", avg_val_loss, epoch)
                logger.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

                print(
                    f"\nEpoch {epoch + 1}/{self.num_epoch}: "
                    f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}\n"
                    f"  Train Dice - Mean: {dice_train_mean:.4f}, BG: {dice_train_bg:.4f}, FG: {dice_train_fg:.4f}\n"
                    f"  Val Dice   - Mean: {dice_val_mean:.4f}, BG: {dice_val_bg:.4f}, FG: {dice_val_fg:.4f}\n"
                    f"  Train HD95 - Mean: {hd95_train_mean:.4f}, BG: {hd95_train_bg:.4f}, FG: {hd95_train_fg:.4f}\n"
                    f"  Val HD95   - Mean: {hd95_val_mean:.4f}, BG: {hd95_val_bg:.4f}, FG: {hd95_val_fg:.4f}"
                )
                
                if avg_batch_latency:
                    print(f"  Avg Batch Latency: {avg_batch_latency:.4f} sec")
                if self.peak_memory_bytes:
                    print(f"  Max Peak Memory: {max_peak_memory_mb:.2f} MB")
                if total_epoch_flops > 0:
                    print(f"  Total Training FLOPs: {total_epoch_flops / 1e12:.2f} TFLOPs")

                # Save checkpoint based on best val_dice_fg
                if dice_val_fg > self.best_val_dice_fg:
                    self.best_val_dice_fg = dice_val_fg
                    save_checkpoint(self.model, optimizer, epoch, log_dir, model_name)
                    print(f" New best val_dice_fg: {self.best_val_dice_fg:.4f} - Checkpoint saved!")
                
                # Also save periodic checkpoints at specific epochs (1, 10, 20, 30, 40, 50, 60, ...)
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    checkpoint_path = log_dir / f"{model_name}_epoch_{epoch + 1}.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"  Periodic checkpoint saved: epoch {epoch + 1}")

        # Save performance metrics
        self._save_performance_metrics(log_dir)

        save_model(self.model, self.kfold_idx)
        torch.save(self.model.state_dict(), log_dir / f"{model_name}.th")
        print(f"Model saved to {log_dir / f'{model_name}.th'}")

    def test_model(self):
        print("Running Patch-based Test...")
        if self.model_name == "UNet2D":
            test_data = load_image_data(
                dataset_path=self.test_data_dir,
                label_subdir='label_gtvp',
                batch_size=self.batch_size,
                shuffle=False,
                is_patches=False
        
            )
        else:
          
            test_data = load_image_data(
                dataset_path=self.train_data_dir,
                label_subdir='label_gtvp',
                batch_size=self.batch_size,
                shuffle=False,
                transform_pipeline="aug_val",
                is_label_sampler=False,
           
            )

        dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
        hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)

        # Add DiceMetricsWrapper for mean computation
        if self.model_name == "UNet2D":
            test_metrics_wrapper = DiceMetricsWrapper(is_2d=True)
        else:
            test_metrics_wrapper = DiceMetricsWrapper()

        post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        post_label = AsDiscrete(to_onehot=self.num_classes)

        self.model.eval()
        dice_metric.reset()
        hd95_metric.reset()
        test_metrics_wrapper.reset()

        with torch.inference_mode():
            test_loader = tqdm.tqdm(test_data, desc="Testing patches")

            for batch in test_loader:
                if self.model_name == "UNet2D":
                    img = batch['image'].to(self.device)
                    mask = batch['label'].to(self.device)
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(1)
                else:
                    img = batch['image']['data'].to(self.device)
                    mask = batch['label']['data'].to(self.device)
                    if mask.dim() == 4:
                        mask = mask.unsqueeze(1)

                logits = self.model(img)

                val_outputs_list = decollate_batch(logits)
                val_labels_list = decollate_batch(mask)

                val_output_convert = [post_pred(i) for i in val_outputs_list]
                val_label_convert = [post_label(i) for i in val_labels_list]

                # Update MONAI metrics for per-class
                dice_metric(y_pred=val_output_convert, y=val_label_convert)
                hd95_metric(y_pred=val_output_convert, y=val_label_convert)
                
                # Update DiceMetricsWrapper for mean values
                if self.model_name == "UNet2D":
                    test_metrics_wrapper.update(logits, batch['label'].to(self.device))
                else:
                    test_metrics_wrapper.update(logits, batch['label']['data'].to(self.device))

        # Aggregate results - MONAI for per-class values
        dice_per_class = dice_metric.aggregate()  # Shape: [num_classes]
        hd95_per_class = hd95_metric.aggregate()  # Shape: [num_classes]
        
        # Extract per-class metrics
        dice_bg = dice_per_class[0].item()
        dice_fg = dice_per_class[1].item()
        
        hd95_bg = hd95_per_class[0].item()
        hd95_fg = hd95_per_class[1].item()
        
        # Get mean values from DiceMetricsWrapper
        test_results = test_metrics_wrapper.compute_all()
        dice_mean = test_results['mean_dice']
        hd95_mean = test_results['hd95']

        print("\nTest Results (Patch-level):")
        print(f"  Dice - Mean: {dice_mean:.4f}, BG: {dice_bg:.4f}, FG: {dice_fg:.4f}")
        print(f"  HD95 - Mean: {hd95_mean:.4f}, BG: {hd95_bg:.4f}, FG: {hd95_fg:.4f}")
        
        # Save to CSV
        test_metrics = {
            "dice_mean": dice_mean,
            "dice_background": dice_bg,
            "dice_foreground": dice_fg,
            "hd95_mean": hd95_mean,
            "hd95_background": hd95_bg,
            "hd95_foreground": hd95_fg,
        }
     
      
        
    def evaluate_patient_level(self):
        """
        Performs patient-level evaluation using Sliding Window Inference.
        Loads FULL Volumes instead of Patches.
        """
        print("\nStarting Patient-Level Evaluation (Sliding Window)...")

        if self.model_name == "UNet2D":
            print("Not Applicable for 2D model.")
            return

        # Load FULL Volumes (use_queue=False)
        val_data_full = load_image_data(
            dataset_path=self.val_data_dir,
            label_subdir='label_gtvp',
            batch_size=1,
            shuffle=False,
            is_patches=True,
            patch_size=(64, 64, 64),
            use_queue=False,
            is_label_sampler=False,
            transform_pipeline="aug_val"
        )

        test_data_full = load_image_data(
            dataset_path=self.test_data_dir,
            label_subdir='label_gtvp',
            batch_size=1,
            shuffle=False,
            is_patches=True,
            patch_size=(64, 64, 64),
            use_queue=False,
            is_label_sampler=False,
            transform_pipeline="aug_val"
        )

        # Initialize metrics
        dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
        hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH, get_not_nans=False)
        
        # Add DiceMetricsWrapper for mean computation
        patient_metrics_wrapper = DiceMetricsWrapper()

        post_pred = AsDiscrete(argmax=True, to_onehot=self.num_classes)
        post_label = AsDiscrete(to_onehot=self.num_classes)

        
        def run_split(dataloader, split_name):
            dice_scores_bg = []
            dice_scores_fg = []
            hd95_scores_bg = []
            hd95_scores_fg = []
            patient_metrics = []
            self.model.eval()

            with torch.inference_mode():
                for patient_idx, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Eval {split_name}")):
                    # (B, C, H, W, D)
                    img = batch['image']['data'].to(self.device)
                    label = batch['label']['data'].to(self.device)

                    if img.dim() == 4:
                        img = img.unsqueeze(0)
                    if label.dim() == 4:
                        label = label.unsqueeze(0)

                    # Sliding Window Inference
                    logits = sliding_window_inference(
                        inputs=img,
                        roi_size=(64, 64, 64),
                        sw_batch_size=4,
                        predictor=self.model,
                        overlap=0.5,
                        mode="gaussian"
                       
                    )

                    # Metric Update
                    val_outputs_list = decollate_batch(logits)
                    val_labels_list = decollate_batch(label)

                    val_output_convert = [post_pred(i) for i in val_outputs_list]
                    val_label_convert = [post_label(i) for i in val_labels_list]

                    # Compute per-sample metric immediately
                    dice_metric(y_pred=val_output_convert, y=val_label_convert)
                    hd95_metric(y_pred=val_output_convert, y=val_label_convert)
                    
                    # Update DiceMetricsWrapper for mean computation
                    patient_metrics_wrapper.reset()
                    patient_metrics_wrapper.update(logits, label)
                    patient_wrapper_results = patient_metrics_wrapper.compute_all()

                    # Store result and reset for next patient
                    dice_per_class = dice_metric.aggregate()  # Shape: [num_classes]
                    hd95_per_class = hd95_metric.aggregate()  # Shape: [num_classes]
                    
                    dice_bg = dice_per_class[0].item()
                    dice_fg = dice_per_class[1].item()
                    
                    hd95_bg = hd95_per_class[0].item()
                    hd95_fg = hd95_per_class[1].item()
                    
                    # Get mean values from DiceMetricsWrapper
                    dice_mean = patient_wrapper_results['mean_dice']
                    hd95_mean = patient_wrapper_results['hd95']

                    dice_scores_bg.append(dice_bg)
                    dice_scores_fg.append(dice_fg)
                    hd95_scores_bg.append(hd95_bg)
                    hd95_scores_fg.append(hd95_fg)
                    
                    patient_metrics.append({
                        "patient_id": patient_idx,
                        "dice_mean": dice_mean,
                        "dice_background": dice_bg,
                        "dice_foreground": dice_fg,
                        "hd95_mean": hd95_mean,
                        "hd95_background": hd95_bg,
                        "hd95_foreground": hd95_fg,
                    })

                    dice_metric.reset()
                    hd95_metric.reset()

            mean_dice_bg = np.mean(dice_scores_bg) if dice_scores_bg else float("nan")
            mean_dice_fg = np.mean(dice_scores_fg) if dice_scores_fg else float("nan")
            mean_dice = (mean_dice_bg + mean_dice_fg) / 2
            
            mean_hd95_bg = np.mean(hd95_scores_bg) if hd95_scores_bg else float("nan")
            mean_hd95_fg = np.mean(hd95_scores_fg) if hd95_scores_fg else float("nan")
            mean_hd95 = (mean_hd95_bg + mean_hd95_fg) / 2
            
            print(f"[{split_name}] Dice - Mean: {mean_dice:.4f}, BG: {mean_dice_bg:.4f}, FG: {mean_dice_fg:.4f}")
            print(f"[{split_name}] HD95 - Mean: {mean_hd95:.4f}, BG: {mean_hd95_bg:.4f}, FG: {mean_hd95_fg:.4f}")
            
         
        print("--> Evaluating Validation Set (Full Volume)...")
        val_dice, val_hd95, val_dice_bg, val_dice_fg, val_hd95_bg, val_hd95_fg = run_split(val_data_full, "Validation")

        print("--> Evaluating Test Set (Full Volume)...")
        test_dice, test_hd95, test_dice_bg, test_dice_fg, test_hd95_bg, test_hd95_fg = run_split(test_data_full, "Test")

        print("\nFinal Patient-Level Results:")
        print(f"Validation: Dice={val_dice:.4f}, HD95={val_hd95:.4f}")
        print(f"Test:       Dice={test_dice:.4f}, HD95={test_hd95:.4f}")
        
        # Save summary statistics
        summary_metrics = [
            {
                "split": "validation",
                "dice_mean": val_dice,
                "dice_background": val_dice_bg,
                "dice_foreground": val_dice_fg,
                "hd95_mean": val_hd95,
                "hd95_background": val_hd95_bg,
                "hd95_foreground": val_hd95_fg,
            },
            {
                "split": "test",
                "dice_mean": test_dice,
                "dice_background": test_dice_bg,
                "dice_foreground": test_dice_fg,
                "hd95_mean": test_hd95,
                "hd95_background": test_hd95_bg,
                "hd95_foreground": test_hd95_fg,
            }
        ]
        summary_df = pd.DataFrame(summary_metrics)
        summary_path = Path(self.exp_dir) / "patient_level_summary_metrics.csv"
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
  
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="UNet3D") 
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--lr", type=float, default=0.009)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--train_data_dir", type=str, default="../data/train")
    parser.add_argument("--val_data_dir", type=str, default="../data/val")
    parser.add_argument("--test_data_dir", type=str, default="../data/test")
    parser.add_argument("--test_only", action='store_true') 
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--kfold_idx", type=int, default=0)
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use mixed precision training")
    parser.add_argument("--no_amp", dest='use_amp', action='store_false', help="Disable mixed precision training")
    parser.add_argument("--use_checkpoint", action='store_true', help="Use model checkpoint for testing")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to model checkpoint for testing")

    args = parser.parse_args()
    trainer = TrainMedicalSeg(**vars(args))
    
    if args.test_only:
        if args.use_checkpoint:
            trainer.model = load_model(trainer.model_name, in_channels=trainer.in_channels, num_classes=trainer.num_classes) 
            checkpoint = torch.load(args.checkpoint_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            trainer.model = load_model(trainer.model_name, in_channels=trainer.in_channels, num_classes=trainer.num_classes) 
            model_path = Path(trainer.exp_dir) / f"{trainer.model_name}.th" 
            if not model_path.exists(): model_path = Path(f"{trainer.model_name}.th")
            trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device), strict=False)
            print(f"Loaded model weights from {model_path}")
        trainer.model = trainer.model.to(trainer.device)
        trainer.test_model()
        trainer.evaluate_patient_level()
    else:
        trainer.train_model()
        trainer.test_model()
        #trainer.evaluate_patient_level()

# Reference: 

#https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb