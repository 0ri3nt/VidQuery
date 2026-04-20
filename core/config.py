import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    fps: int = int(os.getenv("FPS", "1"))
    audio_sr: int = int(os.getenv("AUDIO_SR", "16000"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "15"))

    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    ava_dir: Path = Path(os.getenv("AVA_DIR", "data/ava"))

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    hf_token: str = os.getenv("HF_TOKEN", "")

    def validate_neo4j_config(self) -> None:
        missing = []
        if not self.neo4j_uri.strip():
            missing.append("NEO4J_URI")
        if not self.neo4j_user.strip():
            missing.append("NEO4J_USER")
        if not self.neo4j_password.strip():
            missing.append("NEO4J_PASSWORD")

        if missing:
            raise ValueError(f"Missing required Neo4j settings: {', '.join(missing)}")


def get_settings() -> Settings:
    return Settings()
