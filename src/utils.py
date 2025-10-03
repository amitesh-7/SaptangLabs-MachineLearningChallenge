"""
Utility functions and configurations for the reasoning system
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Configuration
CONFIG = {
    "solver_timeout": 10,  # seconds
    "confidence_threshold": 0.7,
    "cv_folds": 5,
    "augmentation_ratio": 2.0,
    "min_solver_agreement": 2,
    "random_seed": 42,
}

# Problem types
PROBLEM_TYPES = [
    "arithmetic",
    "logic",
    "constraint",
    "set_operations",
    "comparison",
    "counting",
    "other"
]


def setup_logging(log_file: Optional[str] = None):
    """Configure logging for the application"""
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # File handler
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level="DEBUG",
            rotation="10 MB"
        )


def load_data(file_path: Path) -> pd.DataFrame:
    """Load CSV data file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path.name}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def save_data(df: pd.DataFrame, file_path: Path):
    """Save DataFrame to CSV"""
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} rows to {file_path.name}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        raise


def save_json(data: Any, file_path: Path):
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON to {file_path.name}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise


def load_json(file_path: Path) -> Any:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {file_path.name}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise


def save_model(model: Any, file_path: Path):
    """Save model using pickle"""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {file_path.name}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise


def load_model(file_path: Path) -> Any:
    """Load model from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {file_path.name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text"""
    import re
    pattern = r'-?\d+\.?\d*'
    numbers = re.findall(pattern, text)
    return [float(n) for n in numbers]


def calculate_confidence(solver_results: List[Dict[str, Any]]) -> float:
    """
    Calculate confidence score based on solver agreement
    
    Args:
        solver_results: List of solver outputs with their results
        
    Returns:
        Confidence score between 0 and 1
    """
    if not solver_results:
        return 0.0
    
    # Count agreements
    answers = [r.get('answer') for r in solver_results if r.get('success')]
    if not answers:
        return 0.0
    
    # Most common answer
    from collections import Counter
    counter = Counter(str(a) for a in answers)
    most_common_count = counter.most_common(1)[0][1]
    
    # Agreement ratio
    agreement_ratio = most_common_count / len(answers)
    
    # Adjust by solver quality
    avg_solver_confidence = sum(r.get('confidence', 0.5) for r in solver_results) / len(solver_results)
    
    return (agreement_ratio + avg_solver_confidence) / 2


def format_answer(answer: Any) -> str:
    """Format answer for output"""
    if isinstance(answer, bool):
        return str(answer)
    elif isinstance(answer, (int, float)):
        # Remove unnecessary decimals
        if isinstance(answer, float) and answer.is_integer():
            return str(int(answer))
        return str(answer)
    else:
        return str(answer).strip()
