"""
Unit tests for Dataset Ingestion Component.
Tests validation of file type, size bounds, encodings, and malformed files.
"""

import os
from pathlib import Path
import pytest
import pandas as pd

from tools.ingestion import DatasetIngestor, IngestionError, IngestionResult


@pytest.fixture
def ingestor(tmp_path):
    """Creates a temporary DatasetIngestor instance."""
    return DatasetIngestor(session_id="test_ingest_session", max_size_mb=10)


def test_valid_csv_ingestion(ingestor, tmp_path):
    """Verifies that a well-formed CSV is properly parsed and stored."""
    csv_file = tmp_path / "valid_data.csv"
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "city": ["New York", "London", "Paris", "Tokyo"],
        "target": [0, 1, 0, 1]
    })
    df.to_csv(csv_file, index=False)

    result = ingestor.ingest_csv(csv_file)

    assert isinstance(result, IngestionResult)
    assert result.row_count == 4
    assert result.col_count == 3
    assert "age" in result.columns
    assert "city" in result.columns
    assert Path(result.stored_path).exists()
    assert len(result.preview_rows) == 4


def test_nonexistent_file_rejection(ingestor, tmp_path):
    """Verifies that a missing file path raises IngestionError."""
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(IngestionError) as excinfo:
        ingestor.ingest_csv(missing_file)
    assert "does not exist" in str(excinfo.value)


def test_unsupported_extension_rejection(ingestor, tmp_path):
    """Verifies that non-CSV extensions are rejected."""
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("col1,col2\n1,2", encoding="utf-8")
    with pytest.raises(IngestionError) as excinfo:
        ingestor.ingest_csv(txt_file)
    assert "Unsupported file format" in str(excinfo.value)


def test_empty_file_rejection(ingestor, tmp_path):
    """Verifies that a 0-byte file is rejected."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(IngestionError) as excinfo:
        ingestor.ingest_csv(empty_file)
    assert "completely empty" in str(excinfo.value)


def test_header_only_file_rejection(ingestor, tmp_path):
    """Verifies that a CSV with headers but no data rows is rejected."""
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("feature_a,feature_b\n", encoding="utf-8")
    with pytest.raises(IngestionError) as excinfo:
        ingestor.ingest_csv(header_only)
    assert "contains a header but no data rows" in str(excinfo.value)


def test_file_size_limit_rejection(tmp_path):
    """Verifies that files exceeding the maximum size threshold are rejected."""
    small_limit_ingestor = DatasetIngestor(session_id="test_size", max_size_mb=1)
    large_file = tmp_path / "large.csv"
    # Create 2MB file
    with open(large_file, "wb") as f:
        f.write(b"a" * (2 * 1024 * 1024))
    with pytest.raises(IngestionError) as excinfo:
        small_limit_ingestor.ingest_csv(large_file)
    assert "exceeds maximum allowed limit" in str(excinfo.value)


def test_encoding_fallback_handling(ingestor, tmp_path):
    """Verifies that non-UTF-8 encodings (e.g. latin-1) parse cleanly."""
    latin1_file = tmp_path / "latin1_data.csv"
    # Write characters specific to latin-1
    content = "id,name,value\n1,René,100\n2,Müller,200\n"
    latin1_file.write_bytes(content.encode("latin-1"))

    result = ingestor.ingest_csv(latin1_file)
    assert result.row_count == 2
    assert "René" in [row["name"] for row in result.preview_rows]
