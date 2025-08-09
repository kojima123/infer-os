"""
Device detection and provider selection for Infer-OS runtime optimization.

This module provides automatic detection of available execution providers
and intelligent selection based on workload characteristics.
"""

import onnxruntime as ort
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def get_available_providers() -> List[str]:
    """Get list of available ONNX Runtime execution providers."""
    try:
        return ort.get_available_providers()
    except Exception as e:
        logger.warning(f"Failed to get available providers: {e}")
        return ["CPUExecutionProvider"]

def pick_providers(for_stage: str = "decode") -> List[str]:
    """
    Select optimal execution providers based on workload stage.
    
    Args:
        for_stage: Either "decode" (sequential) or "prefill" (parallel)
        
    Returns:
        List of providers in priority order
    """
    available = get_available_providers()
    logger.info(f"Available providers: {available}")
    
    # NPU candidates (names vary by implementation/environment)
    npu_providers = [p for p in available if any(k in p for k in [
        "Vitis", "AMDNPU", "XDNA", "QNN", "VitisAI"
    ])]
    
    # GPU providers
    rocm_providers = [p for p in available if "ROCMExecutionProvider" in p]
    cuda_providers = [p for p in available if "CUDAExecutionProvider" in p]
    
    # CPU fallback
    cpu_providers = ["CPUExecutionProvider"]
    
    if for_stage == "decode" and npu_providers:
        # For sequential decode, prefer NPU
        priority = npu_providers + rocm_providers + cuda_providers + cpu_providers
    elif for_stage == "prefill" and (rocm_providers or cuda_providers):
        # For parallel prefill, prefer GPU
        priority = rocm_providers + cuda_providers + npu_providers + cpu_providers
    else:
        # Default priority
        priority = rocm_providers + cuda_providers + npu_providers + cpu_providers
    
    # Filter to only available providers
    result = [p for p in priority if p in available]
    logger.info(f"Selected provider priority for {for_stage}: {result}")
    
    return result

def device_string_from_provider(provider: str) -> str:
    """
    Convert provider name to device string for tensor operations.
    
    Args:
        provider: ONNX Runtime provider name
        
    Returns:
        Device string compatible with PyTorch/tensor operations
    """
    if "CUDA" in provider:
        return "cuda"
    elif "ROCM" in provider:
        return "rocm"
    elif "CPU" in provider:
        return "cpu"
    else:
        # NPU or unknown providers - fallback to CPU for tensor ops
        logger.warning(f"Unknown provider {provider}, falling back to CPU")
        return "cpu"

def get_device_info(provider: str) -> dict:
    """
    Get device information for the given provider.
    
    Args:
        provider: ONNX Runtime provider name
        
    Returns:
        Dictionary with device information
    """
    device_str = device_string_from_provider(provider)
    
    info = {
        "provider": provider,
        "device": device_str,
        "supports_iobinding": True,  # Assume true, will be tested
        "memory_pool": None
    }
    
    # Test IOBinding support
    try:
        # This is a simple test - actual support depends on specific operations
        if "NPU" in provider or "Vitis" in provider:
            info["supports_iobinding"] = False  # Conservative assumption
        elif "CPU" in provider:
            info["supports_iobinding"] = True
        else:
            info["supports_iobinding"] = True
    except Exception:
        info["supports_iobinding"] = False
    
    return info

