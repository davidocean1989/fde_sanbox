import os
from pathlib import Path


def _load_env_file():
    """在 import config 的时候，就运行这个方法，让他记性你一次.env环境变量的读取"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    # env_path = ".env"  # 只在本目录下寻找 .env文件，不再向上找
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


def _require(name: str):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"环境变量{name}未设置。请在项目根目录创建.env"
            f"（可参考 .env.example),或通过 shell 导出改变量"
        )
    return value


SILICONFLOW_API_KEY = _require("SILICONFLOW_API_KEY")

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

SILICONFLOW_MODEL_QWEN72B = "Qwen/Qwen2.5-72B-Instruct"

SILICONFLOW_MODEL_DEEPSEEK = "deepseek-ai/DeepSeek-V3"

SILICONFLOW_MODEL_GLM45AIR = "zai-org/GLM-4.5-Air"

SILICONFLOW_MODEL_QWEN7B = "Qwen/Qwen2.5-7B-Instruct"


PROBES_PATH = Path("probes.json")
OUT_PATH = Path("probes.jsonl")
