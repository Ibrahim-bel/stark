"""
Configuration management for the RAGAS question generation system.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VLMConfig:
    """VLM (Vision Language Model) configuration."""

    enabled: bool
    use_remote: bool
    host: str
    port: str
    model: str
    api_key: Optional[str]
    max_tokens: int = 4096
    prompt: str = "OCR the full page to markdown."
    timeout: int = 90
    scale: float = 2.0
    temperature: float = 0.7

    @property
    def endpoint(self) -> str:
        """Get the full VLM endpoint URL."""
        return f"{self.host}:{self.port}"

    @classmethod
    def from_env(cls) -> "VLMConfig":
        """Create VLM config from environment variables."""
        return cls(
            enabled=os.getenv("USE_VLM", "true").lower() == "true",
            use_remote=os.getenv("REMOTE_VLM", "false").lower() == "true",
            host=os.getenv("VLM_HOST", "localhost"),
            port=os.getenv("VLM_PORT", "8000"),
            model=os.getenv("VLM_MODEL", "pixtral"),
            api_key=os.getenv("VLM_API_KEY"),
            max_tokens=int(os.getenv("VLM_MAX_TOKENS", "4096")),
            prompt=os.getenv("VLM_OCR_PROMPT", "OCR the full page to markdown."),
            timeout=int(os.getenv("VLM_TIMEOUT", "90")),
            scale=float(os.getenv("VLM_SCALE", "2.0")),
            temperature=float(os.getenv("VLM_TEMPERATURE", "0.7")),
        )


@dataclass
class LLMConfig:
    """LLM configuration."""

    base_url: Optional[str]
    api_key: str
    model: str
    max_tokens: int
    temperature: float

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLM config from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        return cls(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        )


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    base_url: Optional[str]
    api_key: str
    model: str

    @classmethod
    def from_env(cls, llm_config: Optional["LLMConfig"] = None) -> "EmbeddingConfig":
        """Create embedding config from environment variables.

        llm_config est optionnel : si fourni, ses valeurs servent de fallback
        pour base_url et api_key si les variables EMBEDDING_* ne sont pas definies.
        """
        fallback_url = llm_config.base_url if llm_config else None
        fallback_key = llm_config.api_key if llm_config else ""
        return cls(
            base_url=os.getenv("EMBEDDING_BASE_URL") or fallback_url,
            api_key=os.getenv("EMBEDDING_API_KEY") or fallback_key,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        )

    def to_ragas_embedding(self):
        """
        Construit un LangchainEmbeddingsWrapper(OpenAIEmbeddings) pret a l'emploi.

        Utilise EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL depuis .env.
        verify=False + trust_env=False pour le proxy Squid / cert auto-signe litellm.

        Retourne None si api_key absent ou en cas d'erreur d'import.
        """
        if not self.api_key:
            return None
        try:
            import httpx
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            return LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    model=self.model,
                    http_client=httpx.Client(verify=False, trust_env=False),
                    http_async_client=httpx.AsyncClient(verify=False, trust_env=False),
                )
            )
        except Exception:
            return None


@dataclass
class ProcessingConfig:
    """Processing configuration."""

    mode: str  # 'single' or 'batch'
    input_dir: Path
    output_dir: Path
    num_questions_global: int
    num_questions_per_doc: int
    min_chunk_tokens: int
    max_chunk_tokens: int
    max_keyphrases: int

    @classmethod
    def from_env(cls, args=None) -> "ProcessingConfig":
        """Create processing config from environment variables and CLI args."""
        # Get defaults from env
        input_dir = os.getenv("INPUT_DIR", "./doc/trl_pdf/trl_pdf")
        output_dir = os.getenv("OUTPUT_DIR", "./outputTRL")
        mode = os.getenv("PROCESS_MODE", "batch")

        # Override with CLI args if provided
        if args:
            input_dir = args.input_dir or input_dir
            output_dir = args.output_dir or output_dir
            mode = args.mode or mode

        num_questions_default = int(os.getenv("NUM_QUESTIONS", "0"))

        # --num-questions CLI flag overrides GLOBAL_NUM_QUESTIONS / NUM_QUESTIONS
        num_questions_global = int(
            os.getenv("GLOBAL_NUM_QUESTIONS", str(num_questions_default))
        )
        if args and getattr(args, "num_questions", None) is not None:
            num_questions_global = args.num_questions

        return cls(
            mode=mode,
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            num_questions_global=num_questions_global,
            num_questions_per_doc=int(
                os.getenv("PER_DOC_NUM_QUESTIONS", str(num_questions_default))
            ),
            min_chunk_tokens=int(os.getenv("MIN_CHUNK_TOKENS", "300")),
            max_chunk_tokens=int(os.getenv("MAX_CHUNK_TOKENS", "4096")),
            max_keyphrases=int(os.getenv("MAX_KEYPHRASES", "10")),
        )


@dataclass
class OutputConfig:
    """Output configuration."""

    save_knowledge_graph: bool
    generate_beir_format: bool
    generate_csv: bool
    generate_global_questions: bool
    generate_per_doc_questions: bool
    per_doc_use_global_kg: bool
    use_kg_agent: bool

    @classmethod
    def from_env(cls) -> "OutputConfig":
        """Create output config from environment variables."""
        return cls(
            save_knowledge_graph=os.getenv("SAVE_KNOWLEDGE_GRAPH", "true").lower()
            == "true",
            generate_beir_format=os.getenv("GENERATE_BEIR_FORMAT", "true").lower()
            == "true",
            generate_csv=os.getenv("GENERATE_CSV", "true").lower() == "true",
            generate_global_questions=os.getenv(
                "GENERATE_GLOBAL_QUESTIONS", "true"
            ).lower()
            == "true",
            generate_per_doc_questions=os.getenv(
                "GENERATE_PER_DOC_QUESTIONS", "false"
            ).lower()
            == "true",
            per_doc_use_global_kg=os.getenv("PER_DOC_USE_GLOBAL_KG", "false").lower()
            == "true",
            use_kg_agent=os.getenv("USE_KG_AGENT", "false").lower() == "true",
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    vlm: VLMConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    processing: ProcessingConfig
    output: OutputConfig

    @classmethod
    def from_env(cls, args=None) -> "AppConfig":
        """Create complete app config from environment variables."""
        llm_config = LLMConfig.from_env()

        return cls(
            vlm=VLMConfig.from_env(),
            llm=llm_config,
            embedding=EmbeddingConfig.from_env(llm_config),
            processing=ProcessingConfig.from_env(args),
            output=OutputConfig.from_env(),
        )
