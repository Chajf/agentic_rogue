"""Runtime configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROGUE_ENV_")

    rogue_binary: Path = Path("/usr/local/bin/rogue")
    data_dir: Path = Path("/var/lib/rogue")
    player_name: str = "rogo-agent"
    rogue_options: str = "noterse,noflush,jump,seefloor,nopassgo,tombstone,inven=overwrite"
    terminal_rows: int = 24
    terminal_columns: int = 80
    quiet_window_seconds: float = 0.04
    action_timeout_seconds: float = 3.0
