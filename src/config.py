import os
from pathlib import Path
import warnings


def load_env_file(env_path: str = None) -> None:
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value


class Config:
    def __init__(self):
        load_env_file()
        self._openai_api_key = None
        self._hf_token = None
        self._hf_user_id = None

    @property
    def openai_api_key(self) -> str:
        if self._openai_api_key is None:
            self._openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not self._openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Set it in .env or environment."
                )
        return self._openai_api_key

    @property
    def hf_token(self) -> str:
        if self._hf_token is None:
            self._hf_token = os.environ.get("HF_TOKEN")
            if not self._hf_token:
                raise ValueError("HF_TOKEN not found. Set it in .env or environment.")
        return self._hf_token

    @property
    def hf_user_id(self) -> str:
        if self._hf_user_id is None:
            self._hf_user_id = os.environ.get("HF_USER_ID")
            if not self._hf_user_id:
                raise ValueError(
                    "HF_USER_ID not found. Set it in .env or environment."
                )
        return self._hf_user_id

    def setup_environment(self) -> None:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["HF_TOKEN"] = self.hf_token

    def validate_credentials(self) -> bool:
        try:
            _ = self.openai_api_key
            _ = self.hf_token
            return True
        except ValueError as e:
            warnings.warn(f"Credential validation failed: {e}")
            return False


config = Config()


def setup_credentials() -> Config:
    config.setup_environment()
    if not config.validate_credentials():
        raise RuntimeError("Failed to validate required credentials")
    return config
