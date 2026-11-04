from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.metadata_service import MetadataService
from backend.app.ingestion.upload_service import UPLOAD_DIR


@pytest.fixture
def uploaded_csv():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    def create(filename: str, contents: bytes) -> tuple[str, bytes]:
        path = UPLOAD_DIR / filename
        path.write_bytes(contents)
        created_paths.append(path)
        return filename, contents

    yield create

    for path in created_paths:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("filename", "contents", "expected_columns", "expected_rows"),
    [
        ("phase36_utf8.csv", "city,comment\nMontréal,crème brûlée\n".encode("utf-8"), ["city", "comment"], 1),
        (
            "phase36_bom_semicolon.csv",
            " name ; amount \nAda;10\nBéla;20\n".encode("utf-8-sig"),
            ["name", "amount"],
            2,
        ),
        ("phase36_short_row.csv", b"name,amount,region\nAda,10,Paris\nBen,20\n", ["name", "amount", "region"], 2),
        ("phase36_cp1252.csv", "city\nCafé\n".encode("cp1252"), ["city"], 1),
    ],
)
def test_csv_variations_are_read_without_modifying_source(
    uploaded_csv, filename, contents, expected_columns, expected_rows
):
    filename, original_bytes = uploaded_csv(filename, contents)

    metadata = MetadataService.get_csv_metadata(filename)
    dataframe = CsvReaderService.read_csv(filename)

    assert metadata["column_names"] == expected_columns
    assert metadata["rows"] == expected_rows
    expected_first_values = {
        "phase36_utf8.csv": "Montr\u00e9al",
        "phase36_bom_semicolon.csv": "Ada",
        "phase36_short_row.csv": "Ada",
        "phase36_cp1252.csv": "Caf\u00e9",
    }
    assert dataframe.iloc[0, 0] == expected_first_values[filename]
    assert (UPLOAD_DIR / filename).read_bytes() == original_bytes


def test_header_whitespace_is_normalized_without_changing_data_values(uploaded_csv):
    filename, original_bytes = uploaded_csv(
        "phase36_header_whitespace.csv",
        b" customer_name ,city\n John , Paris \n",
    )

    dataframe = CsvReaderService.read_csv(filename)

    assert list(dataframe.columns) == ["customer_name", "city"]
    assert dataframe.loc[0, "customer_name"] == " John "
    assert dataframe.loc[0, "city"] == " Paris "
    assert (UPLOAD_DIR / filename).read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("phase36_empty_header.csv", b"name,   \nAda,10\n", "empty or whitespace-only column name"),
        ("phase36_duplicate_header.csv", b" name ,name\nAda,10\n", "duplicate column names"),
        ("phase36_bad_rows.csv", b"name,amount\nAda,10\nBen,20,extra\n", "could not be parsed"),
    ],
)
def test_unsafe_csv_variations_return_clear_errors(uploaded_csv, filename, contents, message):
    filename, original_bytes = uploaded_csv(filename, contents)

    with pytest.raises(ValueError, match=message):
        CsvReaderService.read_csv_metadata(filename)

    assert (UPLOAD_DIR / filename).read_bytes() == original_bytes
