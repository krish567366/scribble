import logging
import os
import random
import numpy as np
import pandas as pd
from typing import Any, Dict
import json
from datetime import datetime

def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_json(data: Dict[str, Any], path: str) -> None:
    """Save dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def load_json(path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def get_git_commit() -> str:
    """Get current git commit hash if available."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return 'unknown'

def timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().isoformat()