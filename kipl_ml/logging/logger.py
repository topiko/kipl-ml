import logging

TQDM_W = 81


class Colors:
    grey = "\x1b[0;37m"
    green = "\x1b[1;32m"
    yellow = "\x1b[1;33m"
    red = "\x1b[1;31m"
    purple = "\x1b[1;35m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    reset = "\x1b[0m"
    blink_red = "\x1b[5m\x1b[1;31m"


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    def __init__(self):
        super().__init__()
        self.auto_colorized = True

        format_prefix = (
            f"{Colors.purple}%(asctime)s{Colors.reset} "
            f"{Colors.blue}%(name)25s{Colors.reset} "
        )

        format_suffix = "%(levelname)s >> %(message)s"

        self.FORMATS = {
            logging.DEBUG: format_prefix + Colors.green + format_suffix + Colors.reset,
            logging.INFO: format_prefix + Colors.grey + format_suffix + Colors.reset,
            logging.WARNING: format_prefix
            + Colors.yellow
            + format_suffix
            + Colors.reset,
            logging.ERROR: format_prefix + Colors.red + format_suffix + Colors.reset,
            logging.CRITICAL: format_prefix
            + Colors.blink_red
            + format_suffix
            + Colors.reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")

        if "kipl_ml." in record.name:
            record.name = "".join(record.name.replace("kipl_ml.", ""))
        if len(record.name) > 25:
            record.name = "…" + record.name[-24:]
        record.levelname = record.levelname[:4]
        return formatter.format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.propagate = False
    # Check if the logger already has handlers to prevent duplicate messages
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(ColorFormatter())

        logger.addHandler(sh)

    return logger
