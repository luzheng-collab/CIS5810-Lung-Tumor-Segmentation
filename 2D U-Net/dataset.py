from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class LungDataset(Dataset):
    def __init__(self, root, seq=None, base_seed: int = 42):
        """
        root examples (fold-level):
          E:/.../Preprocessed_for_2D_Unet/splits/train/fold0/train
          E:/.../Preprocessed_for_2D_Unet/splits/train/fold0/val
          E:/.../Preprocessed_for_2D_Unet/splits/test

        Expected under each root:
          root/image/<ID>/slice.npy
          root/label_gtvp/<ID>/slice.npy

        where <ID> is like 'Lung_014'.
        """
        self.root = Path(root)
        self.seq = seq
        self.base_seed = base_seed
        self.samples = []   # list of (img_path, lbl_path)

        img_root = self.root / "image"
        lbl_root = self.root / "label_gtvp"

        print(f"[LungDataset] root      = {self.root}")
        print(f"[LungDataset] img_root  = {img_root} (exists={img_root.exists()})")
        print(f"[LungDataset] lbl_root  = {lbl_root} (exists={lbl_root.exists()})")

        if not img_root.exists():
            print("[LungDataset] WARNING: image root does not exist; dataset will be empty.")
            return

        # --- This is the key: recursively visit ID folders (Lung_014, Lung_XXX, ...) ---
        img_paths = sorted(img_root.rglob("*.npy"))
        print(f"[LungDataset] Found {len(img_paths)} image .npy files under {img_root}")

        for ip in img_paths:
            # rel path from image root: e.g. 'Lung_014/0.npy'
            rel = ip.relative_to(img_root)
            lp = lbl_root / rel
            if not lp.exists():
                print(f"[WARN] Missing label for {rel} -> {lp}")
                continue
            self.samples.append((ip, lp))

        print(f"[LungDataset] Built {len(self.samples)} image–label slice pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        img = np.load(img_path).astype(np.float32)   # (H, W)
        msk = np.load(lbl_path).astype(np.int32)     # (H, W)

        # imgaug expects (H, W, C)
        img_3c = img[..., None]
        segmap = SegmentationMapsOnImage(msk, shape=img_3c.shape[:2])

        if self.seq is not None:
            det = self.seq.to_deterministic()
            img_3c = det.augment_image(img_3c)
            segmap = det.augment_segmentation_maps([segmap])[0]
            msk = segmap.get_arr()

        # to torch (C, H, W)
        img_t = torch.from_numpy(img_3c.transpose(2, 0, 1))        # (1, H, W)
        msk_t = torch.from_numpy(msk[None, ...].astype(np.int64))  # (1, H, W)

        return img_t, msk_t
