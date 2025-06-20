"""
Dataset preparation module for MedFormer.
This module handles:
- Loading and preprocessing medical image-text pairs
- Data augmentation for medical images
- Dataset validation and cleaning
- Creation of training and validation splits
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImageTextDataset(Dataset):
    """
    Dataset class for medical image-text pairs.
    
    This class handles:
    - Loading medical images and their corresponding text descriptions
    - Applying data augmentation to images
    - Tokenizing text descriptions
    - Managing dataset splits
    
    Attributes:
        image_dir (str): Directory containing medical images
        text_file (str): Path to JSON file containing image-text pairs
        tokenizer: Tokenizer for text processing
        transform: Image transformation pipeline
        max_length (int): Maximum sequence length for text
        split (str): Dataset split ('train' or 'val')
    """
    
    def __init__(
        self,
        image_dir: str,
        text_file: str,
        tokenizer,
        transform=None,
        max_length: int = 512,
        split: str = 'train'
    ):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load image-text pairs
        with open(text_file, 'r') as f:
            self.data = json.load(f)
        
        # Set up image transformations
        if transform is None:
            self.transform = self._get_default_transforms(split)
        else:
            self.transform = transform
    
    def _get_default_transforms(self, split: str) -> A.Compose:
        """
        Get default image transformations based on dataset split.
        
        Args:
            split (str): Dataset split ('train' or 'val')
            
        Returns:
            Albumentations transform pipeline
        """
        if split == 'train':
            return A.Compose([
                A.RandomResizedCrop(384, 384, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
            - image: Processed image tensor
            - input_ids: Tokenized text input IDs
            - attention_mask: Attention mask for text
            - labels: Target text labels
        """
        item = self.data[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']
        
        # Process text
        text = item['text']
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def prepare_dataset(
    data_dir: str,
    output_dir: str,
    tokenizer,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[MedicalImageTextDataset, MedicalImageTextDataset]:
    """
    Prepare training and validation datasets.
    
    Args:
        data_dir (str): Directory containing raw data
        output_dir (str): Directory to save processed data
        tokenizer: Tokenizer for text processing
        test_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    image_dir = os.path.join(data_dir, 'images')
    text_file = os.path.join(data_dir, 'annotations.json')
    
    # Load annotations
    with open(text_file, 'r') as f:
        data = json.load(f)
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )
    
    # Save split data
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f)
    
    # Create datasets
    train_dataset = MedicalImageTextDataset(
        image_dir=image_dir,
        text_file=os.path.join(output_dir, 'train.json'),
        tokenizer=tokenizer,
        split='train'
    )
    
    val_dataset = MedicalImageTextDataset(
        image_dir=image_dir,
        text_file=os.path.join(output_dir, 'val.json'),
        tokenizer=tokenizer,
        split='val'
    )
    
    return train_dataset, val_dataset

def create_data_loaders(
    train_dataset: MedicalImageTextDataset,
    val_dataset: MedicalImageTextDataset,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def validate_dataset(dataset: MedicalImageTextDataset) -> bool:
    """
    Validate the dataset for potential issues.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    try:
        # Check if all images exist
        for item in dataset.data:
            image_path = os.path.join(dataset.image_dir, item['image'])
            if not os.path.exists(image_path):
                logger.error(f"Missing image: {image_path}")
                return False
        
        # Check if all texts are valid
        for item in dataset.data:
            if not isinstance(item['text'], str) or len(item['text']) == 0:
                logger.error(f"Invalid text in item: {item}")
                return False
        
        # Test data loading
        sample = dataset[0]
        required_keys = ['image', 'input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in sample:
                logger.error(f"Missing key in sample: {key}")
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False

def main():
    """
    Main function for dataset preparation.
    Demonstrates usage of the dataset preparation functions.
    """
    # Example usage
    data_dir = "path/to/raw/data"
    output_dir = "path/to/processed/data"
    
    # Load tokenizer (example)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        tokenizer=tokenizer
    )
    
    # Validate datasets
    if not validate_dataset(train_dataset) or not validate_dataset(val_dataset):
        logger.error("Dataset validation failed")
        return
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    logger.info(f"Dataset preparation completed successfully")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

if __name__ == "__main__":
    main()