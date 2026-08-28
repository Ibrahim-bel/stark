"""
Document extraction module using Docling.

Docling is lazily imported so that plain-text formats (.md, .txt) can be
processed even when docling is not installed.
"""

import logging
import os
from pathlib import Path

from config import VLMConfig

# Try importing docling; if unavailable, set a flag
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        VlmPipelineOptions,
        PdfPipelineOptions,
    )
    from docling.datamodel.pipeline_options_vlm_model import (
        ApiVlmOptions,
        ResponseFormat,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling_core.types.doc import ImageRefMode

    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False
    InputFormat = None  # type: ignore
    ApiVlmOptions = None  # type: ignore
    ResponseFormat = None  # type: ignore


# Supported file formats (extension → requires_docling flag)
_PLAIN_TEXT_EXTENSIONS = {".md", ".txt"}

if _DOCLING_AVAILABLE:
    SUPPORTED_FORMATS = {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".pptx": InputFormat.PPTX,
        ".html": InputFormat.HTML,
        ".md": InputFormat.MD,
        ".txt": InputFormat.MD,
        ".asciidoc": InputFormat.ASCIIDOC,
        ".xlsx": InputFormat.XLSX,
    }
else:
    # Without docling, only plain-text formats are supported
    SUPPORTED_FORMATS = {
        ".md": "MD",
        ".txt": "MD",
    }


class DocumentExtractor:
    """Extract content from various document formats."""

    def __init__(self, vlm_config: VLMConfig):
        """
        Initialize the document extractor.

        Args:
            vlm_config: VLM configuration
        """
        self.vlm_config = vlm_config

    @staticmethod
    def get_file_format(file_path: Path) -> InputFormat:
        """
        Determine the input format based on file extension.

        Args:
            file_path: Path to the file

        Returns:
            InputFormat enum value

        Raises:
            ValueError: If file format is not supported
        """
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
            )
        return SUPPORTED_FORMATS[ext]

    def _create_vlm_options(self) -> ApiVlmOptions:
        """Create VLM API options."""
        headers = {}
        if self.vlm_config.api_key:
            headers["Authorization"] = f"Bearer {self.vlm_config.api_key}"

        return ApiVlmOptions(
            url=f"http://{self.vlm_config.endpoint}/v1/chat/completions",
            params=dict(
                model=self.vlm_config.model,
                max_tokens=self.vlm_config.max_tokens,
                skip_special_tokens=False,
            ),
            headers=headers,
            prompt=self.vlm_config.prompt,
            timeout=self.vlm_config.timeout,
            scale=self.vlm_config.scale,
            temperature=self.vlm_config.temperature,
            response_format=ResponseFormat.DOCTAGS,
        )

    def _extract_with_vlm(self, file_path: Path) -> str:
        """
        Extract content using VLM pipeline.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted markdown content
        """
        logging.info("=" * 80)
        logging.info("VLM MODE ENABLED - Attempting VLM extraction...")
        logging.info("=" * 80)

        if self.vlm_config.use_remote:
            logging.info(
                f"Using remote VLM model: {self.vlm_config.model} "
                f"at {self.vlm_config.endpoint}"
            )
        else:
            logging.info("Using local VLM model (Granite Docling)")

        pipeline_options = VlmPipelineOptions(
            enable_remote_services=self.vlm_config.use_remote,
            include_images=True,
        )

        if self.vlm_config.use_remote:
            pipeline_options.vlm_options = self._create_vlm_options()

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                )
            }
        )

        logging.info("Starting VLM document conversion...")
        result = doc_converter.convert(file_path)

        extracted_texts = []
        for page in result.pages:
            if page.predictions and page.predictions.vlm_response:
                extracted_texts.append(page.predictions.vlm_response.text)

        markdown_content = "\n\n".join(extracted_texts)

        logging.info("=" * 80)
        logging.info("✅ VLM EXTRACTION SUCCESS")
        logging.info("=" * 80)
        logging.info(f"Output length: {len(markdown_content)} characters")
        logging.info(f"Preview: {markdown_content[:300]}...")
        logging.info("=" * 80)

        return markdown_content

    def _extract_standard(self, file_path: Path, file_format: InputFormat) -> str:
        """
        Extract content using standard pipeline.

        Args:
            file_path: Path to the file
            file_format: Input format

        Returns:
            Extracted markdown content
        """
        logging.info("Starting standard document extraction...")

        if file_format == InputFormat.PDF:
            artifacts_path = os.environ.get("DOCLING_ARTIFACTS_PATH", None)
            pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
        else:
            doc_converter = DocumentConverter()

        logging.info("Converting document...")
        result = doc_converter.convert(file_path)

        markdown_content = result.document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED
        )

        logging.info("=" * 80)
        logging.info("✅ STANDARD EXTRACTION SUCCESS")
        logging.info("=" * 80)
        logging.info(f"Output length: {len(markdown_content)} characters")
        logging.info(f"Preview: {markdown_content[:300]}...")
        logging.info("=" * 80)

        return markdown_content

    def extract(self, file_path: Path) -> str:
        """
        Extract content from a document.

        Args:
            file_path: Path to the document

        Returns:
            Extracted markdown content

        Raises:
            Exception: If extraction fails
        """
        logging.info(f"Extracting content from: {file_path}")
        logging.info(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Fallback : docling non disponible → lecture directe du fichier texte
        if not _DOCLING_AVAILABLE:
            ext = file_path.suffix.lower()
            text_exts = {".md", ".txt", ".asciidoc", ".html", ".rst"}
            if ext in text_exts:
                logging.info("Docling not available — reading file as plain text.")
                return file_path.read_text(encoding="utf-8", errors="replace")
            else:
                raise RuntimeError(
                    f"Docling is not installed and file type '{ext}' cannot be "
                    "read as plain text. Install docling to process this file."
                )

        file_format = self.get_file_format(file_path)
        logging.info(f"Detected format: {file_format}")

        # Try VLM for PDFs if enabled
        if file_format == InputFormat.PDF and self.vlm_config.enabled:
            try:
                return self._extract_with_vlm(file_path)
            except Exception as e:
                logging.warning(f"❌ VLM extraction failed: {e}")
                logging.info("Falling back to standard extraction...")

        # Standard extraction
        try:
            return self._extract_standard(file_path, file_format)
        except Exception as e:
            logging.error(f"❌ Standard extraction failed: {e}", exc_info=True)
            raise
