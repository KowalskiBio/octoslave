import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".octoslave"
CONFIG_FILE = CONFIG_DIR / "config.json"

BASE_URL = "https://llm.ai.e-infra.cz/v1"
DEFAULT_MODEL = "mistral-small-4"

KNOWN_MODELS = [
    "mistral-small-4",
    "qwen3.5",
    "qwen3.5-122b",
    "qwen3-coder",
    "qwen3-coder-30b",
    "qwen3-coder-next",
    "gpt-oss-120b",
    "deepseek-v3.2",
    "deepseek-v3.2-thinking",
    "kimi-k2.5",
    "llama-4-scout-17b-16e-instruct",
    "gemma4",
    "glm-4.7",
    "glm-5",
    "thinker",
    "coder",
    "agentic",
    "mini",
    "redhatai-scout",
]


def load_config() -> dict:
    config = {
        "api_key": "",
        "base_url": BASE_URL,
        "default_model": DEFAULT_MODEL,
    }
    # Env vars override config file
    if os.environ.get("OCTOSLAVE_API_KEY"):
        config["api_key"] = os.environ["OCTOSLAVE_API_KEY"]
    if os.environ.get("OCTOSLAVE_BASE_URL"):
        config["base_url"] = os.environ["OCTOSLAVE_BASE_URL"]
    if os.environ.get("OCTOSLAVE_MODEL"):
        config["default_model"] = os.environ["OCTOSLAVE_MODEL"]

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            # File values fill in what env vars didn't set
            for key in ("api_key", "base_url", "default_model"):
                if not config[key] and saved.get(key):
                    config[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass

    return config


def save_config(api_key: str, base_url: str = BASE_URL, default_model: str = DEFAULT_MODEL):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {"api_key": api_key, "base_url": base_url, "default_model": default_model}
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
