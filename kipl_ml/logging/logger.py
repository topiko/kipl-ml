import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger
