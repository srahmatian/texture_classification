
#%%
from pathlib import Path
import logging
import logging.config
import sys
from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent

DATA_DIR = Path(BASE_DIR, "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = Path(BASE_DIR, "docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

TESTS_DIR = Path(BASE_DIR, "tests")
TESTS_DIR.mkdir(parents=True, exist_ok=True)

HUB_DIR_PRETRAINED = Path(BASE_DIR, "hub", "pretrained_models")
HUB_DIR_PRETRAINED.mkdir(parents=True, exist_ok=True)

LABELS_STR_To_INT = {"valid": 1, "invalid": 0}
LABELS_INT_To_STR = {1: "valid", 0: "invalid"}


# Logging
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": ("%(levelname)s %(asctime)s "
            "[%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n")
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # pretty formatting
