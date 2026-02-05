import logging
import os
from datetime import datetime

def setup_logger(name, log_dir=None, file_name=None, level=logging.INFO):
    """Setup a logger that prints to console and optionally to file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console Handler
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    # File Handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        if file_name is None:
            file_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        fh = logging.FileHandler(os.path.join(log_dir, file_name))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
