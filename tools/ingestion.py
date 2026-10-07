"""
Dataset Ingestion Component for Autonomous AI Data Scientist.
Validates file types, size bounds, encoding, schema, and creates immutable raw copies.
"""

import os
import shutil
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

from core.config import WORKSPACE_DIR


class IngestionError(Exception):
    """Raised when an uploaded dataset fails validation."""
    pass


class IngestionResult(BaseModel):
    """Metadata summary of a successfully validated and ingested dataset."""
    session_id: str
    original_filename: str
    stored_path: str
    file_size_bytes: int
    row_count: int
    col_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)


class DatasetIngestor:
    """
    Validates and ingests CSV datasets into an isolated session workspace.
    """

    SUPPORTED_EXTENSIONS = {".csv"}
    DEFAULT_MAX_SIZE_MB = 200

    def __init__(self, session_id: str, max_size_mb: int = DEFAULT_MAX_SIZE_MB):
        self.session_id = session_id
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.raw_dir = (WORKSPACE_DIR / session_id / "raw").resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def validate_file_metadata(self, file_path: Path) -> None:
        """Validates path existence, extension, and non-empty size."""
        if not file_path.exists():
            raise IngestionError(f"Dataset file does not exist: {file_path}")

        if not file_path.is_file():
            raise IngestionError(f"Provided path is not a file: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise IngestionError(
                f"Unsupported file format '{ext}'. Only CSV (.csv) files are supported in this phase."
            )

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise IngestionError("The uploaded CSV file is completely empty (0 bytes).")

        if file_size > self.max_size_bytes:
            raise IngestionError(
                f"Dataset size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed limit "
                f"of {self.max_size_bytes / (1024*1024):.0f} MB."
            )

    def _read_csv_with_fallback(self, file_path: Path) -> pd.DataFrame:
        """Attempts reading CSV across multiple common encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_exception = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                return df
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_exception = e
                continue
            except Exception as e:
                raise IngestionError(f"Failed to parse CSV file: {e}")

        raise IngestionError(
            f"Could not parse CSV under supported encodings (UTF-8, Latin-1, CP1252). Error: {last_exception}"
        )

    def ingest_csv(self, source_path: Union[str, Path]) -> IngestionResult:
        """
        Validates the CSV, verifies schema non-emptiness, saves an immutable copy,
        and extracts baseline ingestion metadata.
        """
        file_path = Path(source_path).resolve()
        self.validate_file_metadata(file_path)

        df = self._read_csv_with_fallback(file_path)

        # Basic schema integrity validation
        if df.shape[0] == 0:
            raise IngestionError("The CSV contains a header but no data rows.")

        if df.shape[1] == 0:
            raise IngestionError("The CSV contains no columns.")

        # Persist raw immutable copy in session raw directory
        target_path = self.raw_dir / file_path.name
        shutil.copy2(file_path, target_path)

        # Generate sample preview safely converting NaNs to None for JSON compatibility
        preview_df = df.head(5).copy()
        preview_records = preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records")

        dtypes_map = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

        return IngestionResult(
            session_id=self.session_id,
            original_filename=file_path.name,
            stored_path=str(target_path),
            file_size_bytes=file_path.stat().st_size,
            row_count=int(df.shape[0]),
            col_count=int(df.shape[1]),
            columns=[str(col) for col in df.columns],
            dtypes=dtypes_map,
            preview_rows=preview_records
        )
