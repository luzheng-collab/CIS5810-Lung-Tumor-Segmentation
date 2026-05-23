from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import nibabel as nib
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import default_collate

def get_transform(transform_pipeline: str):
        """
        Creates a pipeline for processing data.
        """
        if transform_pipeline == "default":
            return None 
        
        if transform_pipeline == "aug_val":
            return tio.Compose([
                tio.Clamp(out_min=-1000, out_max=400, include=['image']),
                tio.ZNormalization(include=['image']),
            ])

        elif transform_pipeline == "aug_train":
        
            return tio.Compose([
                tio.Clamp(out_min=-1000, out_max=400, include=['image']),
                tio.ZNormalization(include=['image']),
                
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=(-10, 10),
                    isotropic=True,
                    image_interpolation='linear',
                    label_interpolation='nearest',
                    include=['image']
                ),
                tio.RandomFlip(axes=(0, 1, 2), include=['image']
                                 
                )
            ])
            
        return None

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_patches=True, label_subdir='label_gtvp'):
        """
        Medical Image Dataset for NIfTI files.
        
        Args:
            data_dir: Path to directory containing 'image' and label subdirectories
            transform: Optional transforms to apply
    
            resize: Target size (height, width) for resizing, default (128, 128)
            label_subdir: Subdirectory name for labels (e.g., 'label_lung', 'label_heart', 'label_gtvp')
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_subdir = label_subdir    
        self.image_dir = self.data_dir / 'image'
        self.label_dir = self.data_dir / label_subdir
        self.is_patches = is_patches
    
        
    
        self.image_files = sorted(self.image_dir.glob("Lung_*_0000.nii.gz"))

        if is_patches:
            self.preprocess = tio.Compose([
                tio.ToCanonical(),  # Convert to RAS orientation
                tio.CropOrPad((256, 256, 200))
              
            ])

            self.subjects = []
            for img_file in self.image_files:
          
                patient_id = img_file.stem.split('_')[1]
                label_file = self.label_dir / f"Lung_{patient_id}.nii.gz"
                
                
                if label_file.exists():
                    subject = tio.Subject(
                        image=tio.ScalarImage(str(img_file)),
                        label=tio.LabelMap(str(label_file)),
                        patient_id=patient_id
                    )
                    subject = self.preprocess(subject)
                    subject["image"].data = subject["image"].data 

                    self.subjects.append(subject)
                else:
                    print(f"Warning: Label file not found for {img_file.name}: {label_file}")

        if not is_patches:
            self.subjects = []
            for img_file in self.image_files:
             
                patient_id = img_file.stem.split('_')[1]
                label_file = self.label_dir / f"Lung_{patient_id}.nii.gz"
                
                if label_file.exists():
                    self.subjects.append((img_file, label_file))
                else:
                    print(f"Warning: Label file not found for {img_file.name}: {label_file}")
            
        print(f"Found {len(self.subjects)} valid NIfTI image-label pairs in {self.data_dir}")



    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):

        if not self.is_patches: 
         
            img_path, label_path = self.subjects[idx]
           
            img_nii = nib.load(str(img_path))
            label_nii = nib.load(str(label_path))
    
            image = img_nii.get_fdata() 
            label = label_nii.get_fdata()  
            z = image.shape[2] // 2
            image = image[:, :, z]
            label = label[:, :, z]
       
            image = torch.from_numpy(image).float().unsqueeze(0)  
            label = torch.from_numpy(label).long() 
            
            # Resize
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),  
                size=(128, 128),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (1, 256, 256)

            label = torch.nn.functional.interpolate(
                label.unsqueeze(0).unsqueeze(0).float(),
                size=(128, 128),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()  

            if self.transform:
                transformed = self.transform(image=image, label=label)
                image, label = transformed['image'], transformed['label']
        
            return {'image': image, 'label': label}

        else:
            subject = self.subjects[idx]
            assert subject["image"].orientation == ("R", "A", "S")
            assert subject["label"].orientation == ("R", "A", "S") 

            image = subject['image'].data.float()  
            label = subject['label'].data.long()  

            return {'image': image, 'label': label, 'patient_id': subject['patient_id']}

        
        
def load_image_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 0,
    batch_size: int = 32,
    shuffle: bool = False,
    label_subdir: str = 'label_gtvp',
    is_patches: bool = True,
    patch_size: tuple = (64, 64, 64),
    max_queue_length: int = 16,
    samples_per_volume: int = 4,
    is_label_sampler: bool = True,
    use_queue: bool = True
) -> DataLoader | Dataset:
    """
    Constructs the NIfTI image dataset/dataloader.
    
    Args:
        dataset_path (str): Path to dataset directory (should contain 'image' and label subdirectories)
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers for Queue (use 0 for DataLoader when is_patches=True)
        batch_size (int): batch size
        shuffle (bool): should be true for train and false for val
        is_resize (bool): whether to resize images
        resize (tuple): target size (height, width) for resizing images, default (128, 128)
        label_subdir (str): subdirectory name for labels (e.g., 'label_lung', 'label_heart', 'label_gtvp')
        is_patches (bool): whether to use 3D patch-based training with tio.Queue
        patch_size (tuple): size of 3D patches to extract, e.g., (96, 96, 64)
        max_queue_length (int): maximum number of patches in queue
        samples_per_volume (int): number of patches to sample per volume

    Returns:
        DataLoader or tio.Queue
   
    """
    dataset_path = Path(dataset_path)
    data_aug_transform = get_transform(transform_pipeline)
    print("Data augmentation/processing pipeline:", data_aug_transform)

    print(f"Loading NIfTI data from {dataset_path}")
    print(f"Label subdirectory: {label_subdir}")
    
    # Create the dataset
    if is_patches:
        print("Using 3D patch-based dataset")
        dataset = MedicalImageDataset(
            dataset_path, 
            label_subdir=label_subdir,
            transform=data_aug_transform
        )
    else:
        print("Using 2D slice-based dataset")
        dataset = MedicalImageDataset(
            dataset_path, 
            label_subdir=label_subdir,
            is_patches=False,
            transform=data_aug_transform

        )

    print(f"Loaded {len(dataset)} NIfTI image samples")

    
    if is_patches: 
     
        tio_dataset = tio.SubjectsDataset(dataset.subjects, transform=data_aug_transform)
    
        if not use_queue:
            print("Full Volume Loading")
            def collate_with_label(batch):
                batch = default_collate(batch)
                if 'image' in batch and tio.DATA in batch['image']:
                     batch['image'][tio.DATA] = batch['image'][tio.DATA].float()
                if 'label' in batch and tio.DATA in batch['label']:
                     batch['label'][tio.DATA] = batch['label'][tio.DATA].float()
                return batch
            loader= DataLoader(
                    tio_dataset,
                    batch_size=1,       
                    num_workers=num_workers,
                    shuffle=False,      
                    collate_fn=collate_with_label
                )
            loader.original_subjects = dataset.subjects
            return loader
        
        if is_label_sampler:
                sampler = tio.data.LabelSampler(
                    patch_size=patch_size,
                    label_name="label",  
                    label_probabilities= {0: 0.2, 1: 0.8}
                )
        else :
            print("Using Uniform Sampler for patches")
            sampler = tio.data.UniformSampler(patch_size=patch_size)
        # Creat e the patch queue
        patch_queue = tio.Queue(
            subjects_dataset=tio_dataset,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=num_workers,
        )
        
        if not return_dataloader:
            return patch_queue
        
     
        return DataLoader(
            patch_queue,
            batch_size=batch_size,
            num_workers=0,  
        )

    # Non-patch mode (2D slices)
    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
    )


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