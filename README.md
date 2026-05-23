# Volumetric CT Lung Tumor Segmentation

A fully automated pipeline for 3D lung tumor GTV (GTVp) segmentation from CT scans, benchmarking multiple segmentation architectures — from classic CNNs to Transformer-based models — under a unified preprocessing and evaluation framework.

## Models

| Architecture | Paradigm | Key Features |
|---|---|---|
| **2D U-Net** | Slice-based CNN | Class-balanced sampling, ensemble learning restacked to 3D |
| **3D U-Net** | Volumetric CNN | Sliding-window 3D patch training |
| **UNETR** | CNN–Transformer | ViT encoder + convolutional decoder (Hatamizadeh et al., 2022) |
| **Linear UNETR** | Linear-attention Transformer | $\mathcal{O}(N)$ attention, Transformer encoder + convolutional decoder |

## Pipeline

All models are trained on uniformly preprocessed CT volumes:

1. **Intensity normalization**
2. **Resampling** to common voxel spacing
3. **Lung-focused cropping** with aligned CT–contour pairs
4. **Patch/slice extraction** — 2D: class-balanced slice sampling; 3D: sliding-window overlapping patches

## ⭐ Linear UNETR — Memory-Efficient Transformer for 3D Segmentation

Standard UNETR's multi-head self-attention scales quadratically with sequence length $\mathcal{O}(N^2)$, making it capacity-heavy and memory-intensive for 3D medical volumes. Linear UNETR addresses this with $\mathcal{O}(N)$ attention, enabling larger patch sizes and higher batch sizes under the same GPU budget — a critical advantage for real-world deployment where GPU resources are constrained.

## Evaluation

Performance is assessed using:

| Metric | Description |
|---|---|
| **Dice ↑** | Dice similarity coefficient |
| **IoU ↑** | Intersection over Union |
| **HD95 ↓** | 95th percentile Hausdorff distance |


## Project Structure

```
├── Data.ipynb                    # Data loading & preprocessing
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


## References

- Hatamizadeh, A. et al. (2022). UNETR: Transformers for 3D Medical Image Segmentation. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*.
