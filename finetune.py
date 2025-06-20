#!/usr/bin/env python3
"""
Fine-tuning script for the MedFormer model on medical image-text datasets.
Optimized for the MEDPIX-ClinQA dataset.

This module handles:
- Model fine-tuning on medical image-text pairs
- Training loop implementation
- Validation and evaluation
- Model checkpointing and saving
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from medformer import MedFormer
import random
import numpy as np
import wandb
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataset_prep import MedicalImageTextDataset, prepare_dataset, create_data_loaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MedicalImageTextDataset(Dataset):
    """
    Dataset for medical image-text pairs.
    
    Expected format for the JSON file:
    [
        {
            "image_path": "path/to/image1.jpg",
            "prompt": "What is shown in this image?",
            "response": "This is a chest X-ray showing..."
        },
        ...
    ]
    """
    def __init__(self, json_file, image_dir=None, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file with annotations.
            image_dir (str, optional): Directory with all the images. If None, image paths are assumed to be absolute.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['image_path']
        if self.image_dir:
            image_path = os.path.join(self.image_dir, image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a placeholder image if the actual image can't be loaded
            image = Image.new('RGB', (384, 384), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        prompt = item['prompt']
        response = item['response']
        
        return {
            'image': image,
            'prompt': prompt,
            'response': response
        }

class MedFormerTrainer:
    """
    Trainer class for fine-tuning MedFormer.
    
    This class handles:
    - Training loop implementation
    - Validation and evaluation
    - Model checkpointing
    - Logging and monitoring
    
    Attributes:
        model: MedFormer model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Device to train on
        output_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_epochs=3,
        warmup_steps=100,
        device=None,
        output_dir="./checkpoints",
        gradient_accumulation_steps=1,
        log_steps=10,
        eval_steps=100,
        save_steps=500,
        use_wandb=False,
        project_name="medformer",
        run_name=None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.use_wandb = use_wandb
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize wandb if requested
        if self.use_wandb:
            run_name = run_name or f"medformer-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=project_name, name=run_name, config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "warmup_steps": warmup_steps,
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset) if val_dataset else 0
            })
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self.collate_fn
            )
        else:
            self.val_loader = None
        
        # Set up optimizer and scheduler
        # Only optimize the vision-language adapter and any added tokens
        trainable_params = [
            {'params': model.vision_language_adapter.parameters()},
        ]
        
        # Add token embeddings for any new tokens
        if hasattr(model, 'tokenizer') and hasattr(model.language_model, 'get_input_embeddings'):
            vocab_size = model.language_model.config.vocab_size
            if len(model.tokenizer) > vocab_size:
                # Get the embeddings for the new tokens
                embedding_layer = model.language_model.get_input_embeddings()
                trainable_params.append({
                    'params': embedding_layer.weight[vocab_size:],
                    'lr': learning_rate * 10  # Higher learning rate for new tokens
                })
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Initialize step counter
        self.global_step = 0
    
    def collate_fn(self, batch):
        """
        Custom collate function for the data loader.
        """
        images = [item['image'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        responses = [item['response'] for item in batch]
        
        return {
            'images': images,
            'prompts': prompts,
            'responses': responses
        }
    
    def train(self):
        """
        Train the model.
        """
        logger.info(f"Starting training on device: {self.device}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                images = batch['images']
                prompts = batch['prompts']
                responses = batch['responses']
                
                batch_loss = 0
                
                # Process each example in the batch
                for i in range(len(images)):
                    # Process image
                    image_embeds = self.model.process_image(images[i])
                    
                    # Prepare the prompt with image tokens and expected response
                    full_prompt = f"{self.model.image_start_token}{self.model.image_end_token} {prompts[i]}"
                    full_text = f"{full_prompt} {responses[i]}"
                    
                    # Tokenize the full text
                    inputs = self.model.tokenizer(full_text, return_tensors="pt", truncation=True, 
                                                 max_length=self.model.max_length).to(self.device)
                    input_ids = inputs.input_ids
                    attention_mask = inputs.attention_mask
                    
                    # Find the position of the image tokens
                    image_start_id = self.model.tokenizer.convert_tokens_to_ids(
                        self.model.tokenizer.tokenize(self.model.image_start_token))[0]
                    image_end_id = self.model.tokenizer.convert_tokens_to_ids(
                        self.model.tokenizer.tokenize(self.model.image_end_token))[0]
                    
                    # Find positions of image tokens in the input_ids
                    image_start_pos = (input_ids == image_start_id).nonzero(as_tuple=True)[1][0]
                    image_end_pos = (input_ids == image_end_id).nonzero(as_tuple=True)[1][0]
                    
                    # Remove the image tokens from input_ids and insert the image embeddings
                    prefix_input_ids = input_ids[:, :image_start_pos]
                    suffix_input_ids = input_ids[:, image_end_pos+1:]
                    
                    # Get embeddings for the prefix and suffix
                    prefix_embeds = self.model.language_model.get_input_embeddings()(prefix_input_ids)
                    suffix_embeds = self.model.language_model.get_input_embeddings()(suffix_input_ids)
                    
                    # Concatenate prefix, image, and suffix embeddings
                    inputs_embeds = torch.cat([prefix_embeds, image_embeds, suffix_embeds], dim=1)
                    
                    # Adjust attention mask
                    prefix_attention_mask = attention_mask[:, :image_start_pos]
                    image_attention_mask = torch.ones(image_embeds.shape[0], image_embeds.shape[1], device=self.device)
                    suffix_attention_mask = attention_mask[:, image_end_pos+1:]
                    
                    attention_mask = torch.cat([prefix_attention_mask, image_attention_mask, suffix_attention_mask], dim=1)
                    
                    # Prepare labels (shift input_ids right)
                    labels = torch.cat([
                        torch.full((1, image_embeds.shape[1]), -100, device=self.device),  # Ignore image tokens
                        suffix_input_ids
                    ], dim=1)
                    
                    # Forward pass
                    outputs = self.model.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Calculate loss
                    loss = outputs.loss
                    batch_loss += loss.item()
                    
                    # Backward pass
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                
                # Average batch loss
                batch_loss /= len(images)
                train_loss += batch_loss
                train_steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": batch_loss})
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Log metrics
                    if self.global_step % self.log_steps == 0:
                        if self.use_wandb:
                            wandb.log({"train/loss": batch_loss, "train/lr": self.scheduler.get_last_lr()[0]}, step=self.global_step)
                    
                    # Evaluate
                    if self.val_loader and self.global_step % self.eval_steps == 0:
                        val_loss = self.evaluate()
                        logger.info(f"Step {self.global_step} - Validation loss: {val_loss:.4f}")
                        
                        if self.use_wandb:
                            wandb.log({"val/loss": val_loss}, step=self.global_step)
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_model(os.path.join(self.output_dir, "best_model"))
                            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
                    
                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint-step-{self.global_step}"))
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_steps
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation at the end of each epoch
            if self.val_loader:
                val_loss = self.evaluate()
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                if self.use_wandb:
                    wandb.log({"val/epoch_loss": val_loss}, step=self.global_step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(self.output_dir, "best_model"))
                    logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save checkpoint at the end of each epoch
            self.save_model(os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}"))
        
        # Final save
        self.save_model(os.path.join(self.output_dir, "final_model"))
        logger.info("Training complete!")
        
        if self.use_wandb:
            wandb.finish()
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        val_loss = 0
        val_steps = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in progress_bar:
                images = batch['images']
                prompts = batch['prompts']
                responses = batch['responses']
                
                batch_loss = 0
                
                # Process each example in the batch
                for i in range(len(images)):
                    # Process image
                    image_embeds = self.model.process_image(images[i])
                    
                    # Prepare the prompt with image tokens and expected response
                    full_prompt = f"{self.model.image_start_token}{self.model.image_end_token} {prompts[i]}"
                    full_text = f"{full_prompt} {responses[i]}"
                    
                    # Tokenize the full text
                    inputs = self.model.tokenizer(full_text, return_tensors="pt", truncation=True, 
                                                 max_length=self.model.max_length).to(self.device)
                    input_ids = inputs.input_ids
                    attention_mask = inputs.attention_mask
                    
                    # Find the position of the image tokens
                    image_start_id = self.model.tokenizer.convert_tokens_to_ids(
                        self.model.tokenizer.tokenize(self.model.image_start_token))[0]
                    image_end_id = self.model.tokenizer.convert_tokens_to_ids(
                        self.model.tokenizer.tokenize(self.model.image_end_token))[0]
                    
                    # Find positions of image tokens in the input_ids
                    image_start_pos = (input_ids == image_start_id).nonzero(as_tuple=True)[1][0]
                    image_end_pos = (input_ids == image_end_id).nonzero(as_tuple=True)[1][0]
                    
                    # Remove the image tokens from input_ids and insert the image embeddings
                    prefix_input_ids = input_ids[:, :image_start_pos]
                    suffix_input_ids = input_ids[:, image_end_pos+1:]
                    
                    # Get embeddings for the prefix and suffix
                    prefix_embeds = self.model.language_model.get_input_embeddings()(prefix_input_ids)
                    suffix_embeds = self.model.language_model.get_input_embeddings()(suffix_input_ids)
                    
                    # Concatenate prefix, image, and suffix embeddings
                    inputs_embeds = torch.cat([prefix_embeds, image_embeds, suffix_embeds], dim=1)
                    
                    # Adjust attention mask
                    prefix_attention_mask = attention_mask[:, :image_start_pos]
                    image_attention_mask = torch.ones(image_embeds.shape[0], image_embeds.shape[1], device=self.device)
                    suffix_attention_mask = attention_mask[:, image_end_pos+1:]
                    
                    attention_mask = torch.cat([prefix_attention_mask, image_attention_mask, suffix_attention_mask], dim=1)
                    
                    # Prepare labels (shift input_ids right)
                    labels = torch.cat([
                        torch.full((1, image_embeds.shape[1]), -100, device=self.device),  # Ignore image tokens
                        suffix_input_ids
                    ], dim=1)
                    
                    # Forward pass
                    outputs = self.model.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Calculate loss
                    loss = outputs.loss
                    batch_loss += loss.item()
                
                # Average batch loss
                batch_loss /= len(images)
                val_loss += batch_loss
                val_steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": batch_loss})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps
        
        return avg_val_loss
    
    def save_model(self, output_dir):
        """
        Save the model.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vision-language adapter
        torch.save(self.model.vision_language_adapter.state_dict(), os.path.join(output_dir, "vision_language_adapter.pt"))
        
        # Save tokenizer
        self.model.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config = {
            "vision_model_id": self.model.vision_model.config._name_or_path,
            "language_model_id": self.model.language_model.config._name_or_path,
            "image_start_token": self.model.image_start_token,
            "image_end_token": self.model.image_end_token,
            "max_length": self.model.max_length,
            "num_adapter_layers": self.model.vision_language_adapter.num_layers,
            "num_attention_heads": self.model.vision_language_adapter.cross_attention_layers[0].num_heads
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Save training args
        training_args = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "global_step": self.global_step
        }
        
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(training_args, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MedFormer")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to the training data JSON file")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to the validation data JSON file")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory containing the images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save the model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--vision_model", type=str, 
                        default="google/siglip2-so400m-patch14-384",
                        help="Vision model to use")
    parser.add_argument("--language_model", type=str, 
                        default="MBZUAI/BiMediX2-8B",
                        help="Language model to use")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Number of steps between logging")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="medformer",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--num_adapter_layers", type=int, default=2,
                        help="Number of layers in the vision-language adapter")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="Number of attention heads in cross-attention")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create datasets
    train_dataset = MedicalImageTextDataset(args.train_data, args.image_dir)
    logger.info(f"Created training dataset with {len(train_dataset)} examples")
    
    val_dataset = None
    if args.val_data:
        val_dataset = MedicalImageTextDataset(args.val_data, args.image_dir)
        logger.info(f"Created validation dataset with {len(val_dataset)} examples")
    
    # Initialize model
    model = MedFormer(
        vision_model_id=args.vision_model,
        language_model_id=args.language_model,
        load_in_4bit=not args.no_4bit,
        max_length=args.max_length,
        num_adapter_layers=args.num_adapter_layers,
        num_attention_heads=args.num_attention_heads
    )
    
    # Load adapter if resuming from checkpoint
    if args.resume_from:
        adapter_path = os.path.join(args.resume_from, "vision_language_adapter.pt")
        if os.path.exists(adapter_path):
            model.load_adapter(adapter_path)
            logger.info(f"Resumed training from checkpoint: {args.resume_from}")
    
    # Initialize trainer
    trainer = MedFormerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main() 