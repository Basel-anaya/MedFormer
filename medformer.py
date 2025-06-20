"""
MedFormer: A multimodal medical AI model that combines vision and language models for medical image analysis.
This module implements the core architecture and functionality of MedFormer, including:
- Cross-attention mechanism for vision-language integration
- Vision-language adapter for feature fusion
- Main MedFormer model combining SigLIP2 and Qwen2.5-7B-Instruct
- Image processing and text generation capabilities
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModel,
    BitsAndBytesConfig,
    SiglipImageProcessor
)
from typing import List, Optional, Tuple, Union
import logging
import math

# Configure logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossAttention(nn.Module):
    """
    Implements a cross-attention mechanism for integrating vision and language features.
    
    This class implements scaled dot-product attention where:
    - Vision features are used as queries (Q)
    - Language features are used as keys (K) and values (V)
    
    Attributes:
        num_heads (int): Number of attention heads for multi-head attention
        head_dim (int): Dimension of each attention head
        scale (float): Scaling factor for dot-product attention
        q_proj, k_proj, v_proj (nn.Linear): Projection layers for Q, K, V
        out_proj (nn.Linear): Output projection layer
        dropout (nn.Dropout): Dropout layer for regularization
    """
    def __init__(self, vision_dim, language_dim, num_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Initialize projection layers for query, key, and value
        self.q_proj = nn.Linear(vision_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(language_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(language_dim, num_heads * head_dim)
        
        # Output projection to combine attention heads
        self.out_proj = nn.Linear(num_heads * head_dim, language_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, vision_features, language_features, attention_mask=None):
        """
        Forward pass of the cross-attention mechanism.
        
        Args:
            vision_features: Tensor of shape [batch_size, seq_len_v, vision_dim]
            language_features: Tensor of shape [batch_size, seq_len_l, language_dim]
            attention_mask: Optional mask for attention computation
            
        Returns:
            Tensor of shape [batch_size, seq_len_v, language_dim]
        """
        batch_size = vision_features.shape[0]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(vision_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(language_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(language_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with scaling
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)
        
        return output

class VisionLanguageAdapter(nn.Module):
    """
    Advanced adapter for integrating vision and language models.
    
    This adapter uses a combination of:
    - Cross-attention layers for feature interaction
    - Feed-forward networks for feature processing
    - Gated fusion mechanism for controlled information flow
    - Learnable temperature parameter for attention control
    
    Attributes:
        vision_dim (int): Dimension of vision features
        language_dim (int): Dimension of language features
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of adapter layers
        vision_proj, language_proj (nn.Linear): Initial projection layers
        cross_attention_layers (nn.ModuleList): List of cross-attention layers
        norm_layers (nn.ModuleList): Layer normalization layers
        ff_layers (nn.ModuleList): Feed-forward networks
        gate (nn.Linear): Gating mechanism
        temperature (nn.Parameter): Learnable temperature parameter
    """
    def __init__(self, vision_dim, language_dim, hidden_dim=1024, num_layers=2, num_heads=8):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(hidden_dim, hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim, language_dim)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
    
    def forward(self, vision_features, language_context=None):
        """
        Forward pass through the adapter.
        
        Args:
            vision_features: Tensor of shape [batch_size, seq_len_v, vision_dim]
            language_context: Optional tensor of shape [batch_size, seq_len_l, language_dim]
                             If None, will use a learnable context vector
        
        Returns:
            Tensor of shape [batch_size, seq_len_v, language_dim]
        """
        # Handle single feature vector case
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)
        
        batch_size, seq_len_v, _ = vision_features.shape
        
        # Project vision features
        x = self.vision_proj(vision_features)
        
        # Handle language context
        if language_context is not None:
            context = self.language_proj(language_context)
        else:
            context = torch.zeros(batch_size, 1, self.hidden_dim, device=vision_features.device)
        
        # Process through layers
        for i in range(self.num_layers):
            # Cross-attention
            attn_output = self.cross_attention_layers[i](x, context)
            
            # Residual connection and layer norm
            x = self.norm_layers[i](x + attn_output)
            
            # Feed-forward network
            ff_output = self.ff_layers[i](x)
            
            # Residual connection
            x = x + ff_output
        
        # Apply gating if context is provided
        if language_context is not None:
            gate_input = torch.cat([x, context[:, :seq_len_v]], dim=-1)
            gate_value = torch.sigmoid(self.gate(gate_input))
            x = gate_value * x + (1 - gate_value) * context[:, :seq_len_v]
        
        # Scale by temperature
        x = x * self.temperature
        
        # Final projection
        output = self.final_proj(x)
        
        return output

class MedFormer(nn.Module):
    """
    MedFormer: A multimodal medical AI model combining SigLIP2 vision encoder with Qwen2.5-7B-Instruct.
    
    This class implements the complete MedFormer architecture, including:
    - Vision model (SigLIP2) for image feature extraction
    - Language model (Qwen2.5-7B-Instruct) for text generation
    - Vision-language adapter for feature integration
    - Specialized methods for medical image analysis
    
    Attributes:
        device (str): Device to run the model on
        max_length (int): Maximum sequence length for text generation
        vision_model_id (str): ID of the vision model
        vision_processor (SiglipImageProcessor): Image processor for SigLIP2
        vision_model (AutoModel): SigLIP2 vision encoder
        language_model (AutoModelForCausalLM): Qwen2.5-7B-Instruct model
        tokenizer (AutoTokenizer): Tokenizer for the language model
        vision_language_adapter (VisionLanguageAdapter): Feature integration adapter
        image_start_token, image_end_token (str): Special tokens for image context
    """
    def __init__(
        self, 
        vision_model_id: str = "google/siglip2-so400m-patch14-384",
        language_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = True,
        vision_hidden_dim: int = 1024,
        max_length: int = 2048,
        num_adapter_layers: int = 2,
        num_attention_heads: int = 8
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.vision_model_id = vision_model_id
        
        # Initialize vision model
        logger.info(f"Loading vision model: {vision_model_id}")
        self.vision_processor = SiglipImageProcessor.from_pretrained(vision_model_id)
        self.vision_model = AutoModel.from_pretrained(vision_model_id).to(device)
        self.vision_model.eval()
        
        # Initialize language model with quantization
        logger.info(f"Loading language model: {language_model_id}")
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_id,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16
            )
        else:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_id,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_id)
        
        # Get embedding dimensions
        with torch.no_grad():
            # Get vision embedding dimension
            dummy_image = Image.new('RGB', (384, 384), color='white')
            dummy_inputs = self.vision_processor(images=dummy_image, return_tensors="pt").to(device)
            vision_emb = self.vision_model.get_image_features(**dummy_inputs)
            vision_dim = vision_emb.shape[-1]
            
            # Get language model embedding dimension
            language_dim = self.language_model.config.hidden_size
        
        logger.info(f"Vision embedding dimension: {vision_dim}")
        logger.info(f"Language model embedding dimension: {language_dim}")
        
        # Initialize vision-language adapter
        self.vision_language_adapter = VisionLanguageAdapter(
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=vision_hidden_dim,
            num_layers=num_adapter_layers,
            num_heads=num_attention_heads
        ).to(device)
        
        # Special tokens for image context
        self.image_start_token = "<image>"
        self.image_end_token = "</image>"
        
        # Add special tokens to tokenizer
        special_tokens = {"additional_special_tokens": [self.image_start_token, self.image_end_token]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            logger.info(f"Added {num_added} special tokens to the tokenizer")
            self.language_model.resize_token_embeddings(len(self.tokenizer))
    
    def process_image(self, image):
        """
        Process an image through the vision encoder and adapter.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Tensor of processed image features
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            raise ValueError("Image must be a PIL Image or a path to an image")
        
        # Process image
        inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
        
        # Get vision embeddings
        with torch.no_grad():
            vision_embeds = self.vision_model.get_image_features(**inputs)
            
            # Add a sequence dimension if needed (batch_size, dim) -> (batch_size, 1, dim)
            if len(vision_embeds.shape) == 2:
                vision_embeds = vision_embeds.unsqueeze(1)
        
        # Process through adapter
        adapted_embeds = self.vision_language_adapter(vision_embeds)
        
        return adapted_embeds
    
    def generate_from_image_and_prompt(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ):
        """
        Generate text from an image and a prompt.
        
        Args:
            image: Input image
            prompt: Text prompt to guide generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        # Process image
        image_embeds = self.process_image(image)
        
        # Prepare prompt with image tokens
        full_prompt = f"{self.image_start_token}{self.image_end_token} {prompt}"
        
        # Tokenize prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, 
                               max_length=self.max_length).to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Find image token positions
        image_start_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.image_start_token))[0]
        image_end_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.image_end_token))[0]
        
        # Find positions of image tokens in the input_ids
        image_start_pos = (input_ids == image_start_id).nonzero(as_tuple=True)[1][0]
        image_end_pos = (input_ids == image_end_id).nonzero(as_tuple=True)[1][0]
        
        # Prepare embeddings
        prefix_input_ids = input_ids[:, :image_start_pos]
        suffix_input_ids = input_ids[:, image_end_pos+1:]
        
        # Get embeddings for the prefix and suffix
        with torch.no_grad():
            prefix_embeds = self.language_model.get_input_embeddings()(prefix_input_ids)
            suffix_embeds = self.language_model.get_input_embeddings()(suffix_input_ids)
        
        # Combine embeddings
        inputs_embeds = torch.cat([prefix_embeds, image_embeds, suffix_embeds], dim=1)
        
        # Prepare attention mask
        prefix_attention_mask = attention_mask[:, :image_start_pos]
        image_attention_mask = torch.ones(image_embeds.shape[0], image_embeds.shape[1], device=self.device)
        suffix_attention_mask = attention_mask[:, image_end_pos+1:]
        
        attention_mask = torch.cat([prefix_attention_mask, image_attention_mask, suffix_attention_mask], dim=1)
        
        # Generate text
        with torch.no_grad():
            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs
            )
        
        # Decode and clean up output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt_text):
            generated_text = generated_text[len(prompt_text):].strip()
        
        return generated_text
    
    def answer_medical_question(self, image, question: str, **kwargs):
        """
        Answer medical questions about images.
        
        Args:
            image: Input medical image
            question: Medical question to answer
            **kwargs: Additional generation parameters
            
        Returns:
            Generated answer text
        """
        prompt = f"Analyze the following medical image and answer this question: {question}\n\nDetailed medical analysis:"
        return self.generate_from_image_and_prompt(image, prompt, **kwargs)
    
    def describe_medical_image(self, image, **kwargs):
        """
        Generate detailed medical image descriptions.
        
        Args:
            image: Input medical image
            **kwargs: Additional generation parameters
            
        Returns:
            Generated description text
        """
        prompt = "Provide a detailed medical description of this image, including any visible abnormalities, anatomical structures, and potential diagnoses:"
        return self.generate_from_image_and_prompt(image, prompt, **kwargs)
    
    def save_adapter(self, save_path: str):
        """
        Save the vision-language adapter weights.
        
        Args:
            save_path: Path to save the adapter weights
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.vision_language_adapter.state_dict(), save_path)
        logger.info(f"Saved vision-language adapter to {save_path}")
    
    def load_adapter(self, load_path: str):
        """
        Load the vision-language adapter weights.
        
        Args:
            load_path: Path to load the adapter weights from
        """
        self.vision_language_adapter.load_state_dict(torch.load(load_path, map_location=self.device))
        logger.info(f"Loaded vision-language adapter from {load_path}")

def parse_args():
    """
    Parse command line arguments for the MedFormer CLI.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MedFormer: Multimodal Medical AI")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="Describe this medical image in detail.", 
                        help="Text prompt to guide the model")
    parser.add_argument("--max_new_tokens", type=int, default=512, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for text generation")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--vision_model", type=str, 
                        default="google/siglip2-so400m-patch14-384",
                        help="Vision model to use")
    parser.add_argument("--language_model", type=str, 
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Language model to use")
    parser.add_argument("--no_4bit", action="store_true", 
                        help="Disable 4-bit quantization")
    parser.add_argument("--mode", type=str, choices=["describe", "answer"], default="describe",
                        help="Mode: 'describe' for image description or 'answer' for question answering")
    parser.add_argument("--question", type=str, default=None,
                        help="Question to answer (required if mode is 'answer')")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to pre-trained adapter weights")
    parser.add_argument("--num_adapter_layers", type=int, default=2,
                        help="Number of layers in the vision-language adapter")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="Number of attention heads in the cross-attention mechanism")
    return parser.parse_args()

def main():
    """
    Main entry point for the MedFormer CLI.
    Handles model initialization, argument parsing, and text generation.
    """
    args = parse_args()
    
    # Initialize model
    model = MedFormer(
        vision_model_id=args.vision_model,
        language_model_id=args.language_model,
        load_in_4bit=not args.no_4bit,
        num_adapter_layers=args.num_adapter_layers,
        num_attention_heads=args.num_attention_heads
    )
    
    # Load adapter if specified
    if args.adapter_path and os.path.exists(args.adapter_path):
        model.load_adapter(args.adapter_path)
    
    # Generate text based on mode
    if args.mode == "describe":
        generated_text = model.describe_medical_image(
            image=args.image,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    elif args.mode == "answer":
        if not args.question:
            raise ValueError("Question must be provided when mode is 'answer'")
        generated_text = model.answer_medical_question(
            image=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    
    # Print results
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main() 