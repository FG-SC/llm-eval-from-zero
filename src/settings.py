"""Paths, dataset pins and environment setup shared by every script."""
import hashlib
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TASKS_DIR = ROOT / "tasks"
SAMPLE_FILE = DATA_DIR / "sample_ids.json"

GSM8K = {
    "path": "openai/gsm8k",
    "name": "main",
    "split": "test",
    "revision": "740312add88f781978c0658806c59bc2815b9866",
    "n_rows": 1319,
}


def prepare_environment() -> None:
    """Short in-project HF cache (Windows MAX_PATH) and the Groq key from .env."""
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_task_yaml(task: str) -> dict:
    """A task YAML from tasks/, with lm_eval's '!function' tags kept as plain strings."""
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))
    return yaml.load((TASKS_DIR / f"{task}.yaml").read_text(encoding="utf-8"), Loader=Loader)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
