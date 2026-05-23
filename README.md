# CIS5810 Final Project: Lung Tumor Segmentation

Deep learning-based lung tumor segmentation comparing four UNet-based architectures.

## Models

| Model | Description |
|-------|-------------|
| **2D U-Net** | Standard 2D UNet with slice-wise segmentation |
| **3D U-Net** | Volumetric UNet with 3D convolutions |
| **3D UNETR** | UNet Transformer — hybrid CNN-Transformer for 3D medical images |
| **Linear UNETR** | Lightweight variant with linear attention |

## Structure

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

Each model directory contains self-contained training scripts. Run the corresponding Jupyter notebook or Python script:

```bash
# 2D U-Net
jupyter notebook "2D U-Net/Train.ipynb"

# 3D U-Net
python "3D U-Net/train_models.py"

# 3D UNETR
jupyter notebook "3D UNETR/Train.ipynb"

# Linear UNETR
jupyter notebook "Linear UNETR/03-Train_linear.ipynb"
```
