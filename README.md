# Volumetric CT Lung Tumor Segmentation

A fully automated pipeline for 3D lung tumor GTV (GTVp) segmentation from CT scans, benchmarking multiple segmentation architectures — from classic CNNs to Transformer‑based models — under a unified preprocessing and evaluation framework.

- **2D U-Net** — Slice-based segmentation with class-balanced sampling and ensemble learning restacked to 3D
- **3D U-Net** — Volumetric convolutions with sliding-window patch training
- **UNETR** — Hybrid 3D segmentation architecture (Hatamizadeh et al., 2022) with ViT encoder + convolutional decoder
- **Linear UNETR** — Memory-efficient UNETR variant with linear attention with Transformer encoder + convolutional decoder

## Pipeline

All models are trained on uniformly preprocessed CT volumes:

1. **Intensity normalization**
2. **Resampling** to common voxel spacing
3. **Lung-focused cropping** with aligned CT–contour pairs
4. **Patch/slice extraction** — 2D: class-balanced slice sampling; 3D: sliding-window overlapping patches

## ⭐ Linear UNETR: Memory‑Efficient Transformer for 3D Segmentation

Standard Transformer attention scales quadratically:

$$\mathcal{O}(N^2)$$

Linear attention reduces this to:

$$\mathcal{O}(N)$$

**Benefits**
- Lower peak GPU memory
- Larger 3D patch sizes
- Higher batch sizes
- More stable training on clinical‑scale volumes

Linear UNETR demonstrates that linear attention is a practical path to scaling Transformer‑based 3D segmentation for real‑world deployment.

## Evaluation

Performance is assessed using:

| Metric | Description |
|--------|-------------|
| **Dice** | Dice similarity coefficient |
| **IoU** | Intersection over Union |
| **HD95** | 95th percentile Hausdorff distance |

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
