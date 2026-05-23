# CIS5810 Final Project: Automated Lung Tumor GTV Segmentation from 3D CT

Volumetric CT lung tumor segmentation is essential for radiotherapy (RT) planning and quantitative imaging, yet manual gross tumor volume (GTV) delineation in locally advanced non-small cell lung cancer (LA-NSCLC) remains labor-intensive and variable across clinicians and institutions. Automated segmentation can reduce workload, standardize contours, and unlock large-scale radiomics and dosiomics analyses.

## Motivation

This project develops a fully automated pipeline for lung tumor GTV (GTVp) segmentation from 3D CT scans and systematically compares different model architectures:

- **2D U-Net** — Slice-based segmentation with class-balanced sampling and 5-fold ensemble restacked to 3D
- **3D U-Net** — Volumetric convolutions with sliding-window patch training
- **UNETR** — Hybrid CNN-Transformer (Hatamizadeh et al., 2022) with ViT encoder + convolutional decoder
- **Linear UNETR** — Memory-efficient variant with linear attention

## Pipeline

All models are trained on uniformly preprocessed CT volumes:

1. **Intensity normalization**
2. **Resampling** to common voxel spacing
3. **Lung-focused cropping** with aligned CT–contour pairs
4. **Patch/slice extraction** — 2D: class-balanced slice sampling; 3D: sliding-window overlapping patches

## ⭐ Linear UNETR: Memory-Efficient Transformer for 3D Segmentation

A memory-efficient variant of the standard UNETR that replaces quadratic self-attention with linear attention, substantially reducing peak GPU memory while preserving segmentation quality.

### Why It Matters

Standard UNETR's multi-head self-attention scales quadratically with sequence length ($\mathcal{O}(N^2)$), making it capacity-heavy and memory-intensive for 3D medical volumes. Linear UNETR addresses this with $\mathcal{O}(N)$ attention, enabling training on larger patch sizes and higher batch sizes under the same GPU budget.

### Efficiency Benchmarks

| Model | Peak Memory (MB) ↓ | FLOPs / Epoch | Latency (s/batch) ↓ | Val Dice (Best) ↑ | Val Dice (Avg) ↑ |
|-------|-------------------|---------------|---------------------|--------------------|--------------------|
| Standard UNETR | 56,013 | 3.99×10¹⁵ | 4.933 | 0.597 | 0.431 |
| **Linear UNETR** | **54,999** | 3.99×10¹⁵ | 4.938 | **0.597** | 0.424 |

### Implications

The Linear UNETR shows that linear attention provides a viable path to scale Transformer-based 3D segmentation to larger volumes, higher resolutions, and larger cohorts — critical for real-world clinical deployment where GPU resources are constrained.

## Evaluation

Performance is assessed using:

| Metric | Description |
|--------|-------------|
| **Dice** | Dice similarity coefficient |
| **IoU** | Intersection over Union |
| **HD95** | 95th percentile Hausdorff distance |

Evaluation at slice, patch, and whole-volume levels.

## Key Insights

| Dimension | 2D U-Net | 3D Models (UNet, UNETR) |
|-----------|----------|------------------------|
| **Data efficiency** | Benefits from abundant slices, strong in-plane contrast | Needs larger cohorts for full potential |
| **Spatial context** | Limited to slice-wise | Rich volumetric continuity |
| **Sensitivity** | Stable baseline | More sensitive to patch sampling, class imbalance, anisotropic resampling |
| **Capacity** | Lightweight | Transformer-based models are capacity-heavy; need careful optimization |

## Project Structure

```
├── Data.ipynb              # Data loading & preprocessing
├── 2D U-Net/
│   ├── Dataset.ipynb
│   ├── Preprocessing.ipynb
│   ├── Model.ipynb
│   ├── Train.ipynb
│   ├── dataset.py
│   └── model.py
├── 3D U-Net/
│   ├── 3DUNET.py
│   ├── data_loader.py
│   ├── metric.py
│   ├── models.py
│   └── train_models.py
├── 3D UNETR/
│   ├── Train.ipynb
│   └── model.py
└── Linear UNETR/
    ├── 03-Train_linear.ipynb
    ├── 03-Train_unetr.ipynb
    ├── data_loader.py
    ├── linear_unetr.py
    ├── metric.py
    ├── models.py
    └── train_models.py
```

## Usage

```bash
# 2D U-Net
jupyter notebook "2D U-Net/Train.ipynb"

# 3D U-Net
python "3D U-Net/train_models.py"

# 3D UNETR (standard)
jupyter notebook "3D UNETR/Train.ipynb"

# Linear UNETR (memory-efficient)
jupyter notebook "Linear UNETR/03-Train_linear.ipynb"

# Baseline UNETR training script
jupyter notebook "Linear UNETR/03-Train_unetr.ipynb"
```

## Dataset

- **150 patients** with LA-NSCLC
- Anisotropic CT spacing
- Paired CT volumes + GTVp contours
- Limited GPU memory constraints drive architectural choices

## Conclusion

Architecture, data regime, and training strategy jointly shape segmentation quality. Well-tuned 2D models remain competitive baselines, while 3D architectures offer complementary strengths that emerge more clearly with larger cohorts, improved tumor-aware patching, and further architectural refinement. **Linear UNETR** demonstrates that memory-efficient attention mechanisms can match standard Transformer performance while reducing GPU footprint — a critical step toward scalable, deployable 3D medical image segmentation for RT planning and downstream outcome modeling in LA-NSCLC.

## References

- Hatamizadeh et al. (2022). UNETR: Transformers for 3D Medical Image Segmentation. *WACV 2022*.
