import random
import numpy as np
import torch
import os
import json
import pickle
from typing import Any, Dict, Optional
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def save_results(
    results: Dict,
    save_path: str,
    format: str = 'pickle'
):
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        save_path: Path to save file
        format: Format to save ('pickle', 'json')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format == 'pickle':
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'json':
        # Convert numpy types to native Python types
        results_json = convert_to_json_serializable(results)
        with open(save_path, 'w') as f:
            json.dump(results_json, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Results saved to {save_path}")


def load_results(load_path: str, format: str = 'pickle') -> Dict:
    """
    Load results from file.
    
    Args:
        load_path: Path to load file
        format: Format to load ('pickle', 'json')
        
    Returns:
        Results dictionary
    """
    if format == 'pickle':
        with open(load_path, 'rb') as f:
            results = pickle.load(f)
    elif format == 'json':
        with open(load_path, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return results


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device() -> torch.device:
    """
    Get available device (CUDA or CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_experiment_dir(base_dir: str = "results/experiments") -> str:
    """
    Create directory for experiment with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        **kwargs: Additional data to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    load_path: str,
    device: torch.device
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer
        load_path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {load_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    
    return checkpoint


if __name__ == "__main__":
    # Test helpers
    print("Testing helper functions...")
    
    # Test set_seed
    print("\n" + "="*60)
    set_seed(42)
    
    # Test save/load results
    print("\n" + "="*60)
    test_results = {
        'accuracy': 0.95,
        'loss': 0.12,
        'array': np.array([1, 2, 3]),
        'nested': {
            'value': np.float32(3.14)
        }
    }
    
    os.makedirs('results/test', exist_ok=True)
    save_results(test_results, 'results/test/test_results.pkl', format='pickle')
    loaded_results = load_results('results/test/test_results.pkl', format='pickle')
    print(f"Loaded results: {loaded_results}")
    
    # Test JSON serialization
    save_results(test_results, 'results/test/test_results.json', format='json')
    
    # Test device
    print("\n" + "="*60)
    device = get_device()
    
    # Test time formatting
    print("\n" + "="*60)
    print(f"Format 3661 seconds: {format_time(3661)}")
    print(f"Format 125 seconds: {format_time(125)}")
    print(f"Format 45 seconds: {format_time(45)}")
    
    # Test experiment directory
    print("\n" + "="*60)
    exp_dir = create_experiment_dir('results/test_experiments')
    
    print("\n✓ All tests passed!")