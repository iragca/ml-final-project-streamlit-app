import os

from dotenv import load_dotenv
from loguru import logger
from Pathlib import Path


# Load environment variables from .env file
load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJ_ROOT / "data"
MODELS_DIR = PROJ_ROOT / "models"


logger.info(f"{PROJ_ROOT=}")
