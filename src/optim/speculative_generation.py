"""
Lightweight Speculative Generation for Infer-OS.

This module implements speculative decoding with a small draft model
to accelerate inference while maintaining quality control.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class SpeculativeConfig:
    """Configuration for speculative generation."""
    max_draft_tokens: int = 4  # Maximum tokens to generate speculatively
    acceptance_threshold: float = 0.8  # Threshold for accepting draft tokens
    temperature: float = 1.0  # Sampling temperature
    top_k: int = 50  # Top-k sampling
    top_p: float = 0.9  # Top-p (nucleus) sampling
    draft_temperature: float = 1.2  # Higher temperature for draft model
    quality_control: bool = True  # Enable quality control mechanisms
    fallback_on_reject: bool = True  # Fallback to target model on rejection

class TokenSampler:
    """Optimized token sampling for speculative generation."""
    
    def __init__(self, config: SpeculativeConfig):
        self.config = config
    
    def sample_token(self, logits: torch.Tensor, temperature: float = None, 
                    top_k: int = None, top_p: float = None) -> int:
        """
        Sample token from logits with temperature, top-k, and top-p filtering.
        
        Args:
            logits: Model output logits [vocab_size] or [batch_size, vocab_size]
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            
        Returns:
            Sampled token ID
        """
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p
        
        # Ensure logits is 1D
        if logits.dim() > 1:
            logits = logits.view(-1)
        
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        # Simple top-k filtering
        if top_k > 0 and top_k < logits.size(0):
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            # Create mask for top-k tokens
            mask = torch.full_like(logits, float('-inf'))
            mask[top_k_indices] = top_k_values
            logits = mask
        
        # Simple top-p filtering (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff
            cutoff_mask = cumulative_probs <= top_p
            if cutoff_mask.sum() > 0:
                # Keep at least one token
                cutoff_mask[0] = True
                
                # Apply mask
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits[sorted_indices[cutoff_mask]] = sorted_logits[cutoff_mask]
                logits = filtered_logits
        
        # Sample from the filtered distribution
        try:
            probs = F.softmax(logits, dim=-1)
            # Handle case where all probabilities are 0 (all -inf)
            if torch.isnan(probs).all() or torch.isinf(probs).all():
                # Fallback to uniform sampling
                token = torch.randint(0, logits.size(0), (1,)).item()
            else:
                token = torch.multinomial(probs, num_samples=1).item()
            
            # Ensure token is within valid range
            token = max(0, min(token, logits.size(0) - 1))
            
        except Exception as e:
            # Ultimate fallback
            logger.warning(f"Sampling failed: {e}, using random token")
            token = torch.randint(0, logits.size(0), (1,)).item()
        
        return token
    
    def batch_sample_tokens(self, logits: torch.Tensor, num_samples: int = 1,
                           temperature: float = None) -> List[int]:
        """
        Sample multiple tokens from logits.
        
        Args:
            logits: Model output logits [vocab_size]
            num_samples: Number of tokens to sample
            temperature: Sampling temperature
            
        Returns:
            List of sampled token IDs
        """
        tokens = []
        for _ in range(num_samples):
            token = self.sample_token(logits, temperature)
            tokens.append(token)
        
        return tokens

class ModelInterface(ABC):
    """Abstract interface for models used in speculative generation."""
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional model arguments
            
        Returns:
            Output logits
        """
        pass
    
    @abstractmethod
    def get_cache_size(self) -> int:
        """Get current KV cache size."""
        pass
    
    @abstractmethod
    def clear_cache(self):
        """Clear KV cache."""
        pass

class MockDraftModel(ModelInterface):
    """Mock draft model for testing and demonstration."""
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 512):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cache_size = 0
        
        # Simple linear layer for demonstration
        self.projection = torch.nn.Linear(hidden_size, vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simple forward pass."""
        # Simulate fast draft model inference
        time.sleep(0.001)  # 1ms simulation
        
        # Ensure input_ids are within vocabulary range
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        embeddings = self.embedding(input_ids)
        logits = self.projection(embeddings)
        
        self.cache_size += input_ids.size(1)
        
        # Return logits for the last token position
        if logits.dim() == 3:  # [batch, seq, vocab]
            return logits[:, -1, :]  # [batch, vocab]
        else:
            return logits
    
    def get_cache_size(self) -> int:
        return self.cache_size
    
    def clear_cache(self):
        self.cache_size = 0

class MockTargetModel(ModelInterface):
    """Mock target model for testing and demonstration."""
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 2048):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cache_size = 0
        
        # Larger model simulation
        self.projection = torch.nn.Linear(hidden_size, vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with higher latency."""
        # Simulate slower target model inference
        time.sleep(0.005)  # 5ms simulation
        
        # Ensure input_ids are within vocabulary range
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        embeddings = self.embedding(input_ids)
        logits = self.projection(embeddings)
        
        self.cache_size += input_ids.size(1)
        return logits  # Return full sequence logits
    
    def get_cache_size(self) -> int:
        return self.cache_size
    
    def clear_cache(self):
        self.cache_size = 0

class SpeculativeGenerator:
    """
    Main speculative generation engine.
    
    This class implements the core speculative decoding algorithm:
    1. Generate multiple tokens with fast draft model
    2. Verify tokens with target model in parallel
    3. Accept/reject tokens based on probability comparison
    4. Fallback to target model for rejected tokens
    """
    
    def __init__(self, draft_model: ModelInterface, target_model: ModelInterface,
                 config: SpeculativeConfig = None):
        """
        Initialize speculative generator.
        
        Args:
            draft_model: Fast draft model for speculation
            target_model: Accurate target model for verification
            config: Configuration parameters
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config or SpeculativeConfig()
        self.sampler = TokenSampler(self.config)
        
        self.stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "draft_calls": 0,
            "target_calls": 0,
            "acceptance_rate": 0.0,
            "speedup": 0.0
        }
    
    def _generate_draft_tokens(self, input_ids: torch.Tensor, 
                              num_tokens: int) -> List[int]:
        """
        Generate draft tokens using the draft model.
        
        Args:
            input_ids: Current input sequence
            num_tokens: Number of tokens to generate
            
        Returns:
            List of draft token IDs
        """
        draft_tokens = []
        current_ids = input_ids.clone()
        
        for _ in range(num_tokens):
            # Get logits from draft model
            logits = self.draft_model.forward(current_ids)
            self.stats["draft_calls"] += 1
            
            # Sample token with higher temperature for diversity
            token = self.sampler.sample_token(
                logits, 
                temperature=self.config.draft_temperature
            )
            
            draft_tokens.append(token)
            
            # Append token for next iteration
            current_ids = torch.cat([
                current_ids, 
                torch.tensor([[token]], dtype=current_ids.dtype, device=current_ids.device)
            ], dim=1)
        
        return draft_tokens
    
    def _verify_draft_tokens(self, input_ids: torch.Tensor, 
                           draft_tokens: List[int]) -> Tuple[List[int], int]:
        """
        Verify draft tokens using the target model.
        
        Args:
            input_ids: Original input sequence
            draft_tokens: Draft tokens to verify
            
        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        if not draft_tokens:
            return [], 0
        
        # Create sequence with draft tokens
        draft_tensor = torch.tensor([draft_tokens], dtype=input_ids.dtype, device=input_ids.device)
        extended_ids = torch.cat([input_ids, draft_tensor], dim=1)
        
        # Get target model logits for the entire sequence
        target_logits = self.target_model.forward(extended_ids)
        self.stats["target_calls"] += 1
        
        accepted_tokens = []
        
        # Verify each draft token
        for i, draft_token in enumerate(draft_tokens):
            # Get target model probability for this position
            position_logits = target_logits[:, input_ids.size(1) + i - 1, :]
            target_probs = F.softmax(position_logits, dim=-1)
            target_prob = target_probs[0, draft_token].item()
            
            # Get draft model probability (approximate)
            draft_logits = self.draft_model.forward(extended_ids[:, :input_ids.size(1) + i])
            draft_probs = F.softmax(draft_logits, dim=-1)
            draft_prob = draft_probs[0, draft_token].item()
            
            # Acceptance criterion: target_prob / draft_prob >= threshold
            acceptance_ratio = target_prob / (draft_prob + 1e-8)
            
            if acceptance_ratio >= self.config.acceptance_threshold:
                accepted_tokens.append(draft_token)
                self.stats["accepted_tokens"] += 1
            else:
                # Reject this and all subsequent tokens
                self.stats["rejected_tokens"] += len(draft_tokens) - i
                break
        
        return accepted_tokens, len(accepted_tokens)
    
    def _generate_fallback_token(self, input_ids: torch.Tensor) -> int:
        """
        Generate fallback token using target model when draft is rejected.
        
        Args:
            input_ids: Current input sequence
            
        Returns:
            Generated token ID
        """
        logits = self.target_model.forward(input_ids)
        self.stats["target_calls"] += 1
        
        token = self.sampler.sample_token(logits)
        return token
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                stop_tokens: Optional[List[int]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate tokens using speculative decoding.
        
        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            max_length: Maximum total sequence length
            stop_tokens: List of stop token IDs
            
        Returns:
            Tuple of (generated_sequence, generation_stats)
        """
        if stop_tokens is None:
            stop_tokens = []
        
        start_time = time.perf_counter()
        current_ids = input_ids.clone()
        generated_tokens = []
        
        # Reset stats for this generation
        self.stats.update({
            "total_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "draft_calls": 0,
            "target_calls": 0
        })
        
        while current_ids.size(1) < max_length:
            # Determine how many tokens to generate speculatively
            remaining_tokens = max_length - current_ids.size(1)
            num_draft_tokens = min(self.config.max_draft_tokens, remaining_tokens)
            
            if num_draft_tokens <= 0:
                break
            
            # Generate draft tokens
            draft_tokens = self._generate_draft_tokens(current_ids, num_draft_tokens)
            
            # Verify draft tokens
            accepted_tokens, num_accepted = self._verify_draft_tokens(current_ids, draft_tokens)
            
            if num_accepted > 0:
                # Add accepted tokens
                accepted_tensor = torch.tensor([accepted_tokens], 
                                             dtype=current_ids.dtype, 
                                             device=current_ids.device)
                current_ids = torch.cat([current_ids, accepted_tensor], dim=1)
                generated_tokens.extend(accepted_tokens)
                self.stats["total_tokens"] += num_accepted
                
                # Check for stop tokens
                if any(token in stop_tokens for token in accepted_tokens):
                    break
            
            # If not all tokens were accepted, generate one with target model
            if num_accepted < len(draft_tokens) and current_ids.size(1) < max_length:
                fallback_token = self._generate_fallback_token(current_ids)
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[fallback_token]], dtype=current_ids.dtype, device=current_ids.device)
                ], dim=1)
                generated_tokens.append(fallback_token)
                self.stats["total_tokens"] += 1
                
                if fallback_token in stop_tokens:
                    break
        
        # Calculate final statistics
        generation_time = time.perf_counter() - start_time
        
        if self.stats["total_tokens"] > 0:
            self.stats["acceptance_rate"] = self.stats["accepted_tokens"] / (
                self.stats["accepted_tokens"] + self.stats["rejected_tokens"]
            )
        
        # Estimate speedup (simplified calculation)
        baseline_calls = self.stats["total_tokens"]  # One target call per token
        actual_calls = self.stats["target_calls"]
        if actual_calls > 0:
            self.stats["speedup"] = baseline_calls / actual_calls
        
        generation_stats = {
            **self.stats,
            "generation_time": generation_time,
            "tokens_per_second": self.stats["total_tokens"] / generation_time if generation_time > 0 else 0,
            "generated_tokens": generated_tokens
        }
        
        return current_ids, generation_stats
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats.update({
            "total_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "draft_calls": 0,
            "target_calls": 0,
            "acceptance_rate": 0.0,
            "speedup": 0.0
        })

def create_speculative_generator(draft_model: ModelInterface, 
                               target_model: ModelInterface,
                               max_draft_tokens: int = 4,
                               acceptance_threshold: float = 0.8) -> SpeculativeGenerator:
    """
    Factory function to create speculative generator.
    
    Args:
        draft_model: Fast draft model
        target_model: Accurate target model
        max_draft_tokens: Maximum tokens to generate speculatively
        acceptance_threshold: Threshold for accepting draft tokens
        
    Returns:
        Configured speculative generator
    """
    config = SpeculativeConfig(
        max_draft_tokens=max_draft_tokens,
        acceptance_threshold=acceptance_threshold
    )
    
    return SpeculativeGenerator(draft_model, target_model, config)

