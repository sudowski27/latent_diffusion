"""version 0.1.0"""
import logging


def set_logger(filename: str, level):
    """
    """
    logging.basicConfig(
        filename=filename,  # Log file name
        level=level,   # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    )

    logger = logging.getLogger(__name__)

    return logger
