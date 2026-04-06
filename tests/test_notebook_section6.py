import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "01_data_collection.ipynb"


def _load_cells():
    with NOTEBOOK_PATH.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    return notebook["cells"]


def test_notebook_has_section_6_cells_28_to_34():
    cells = _load_cells()

    assert len(cells) >= 34

    section_markdown = "".join(cells[27]["source"])
    consolidation_code = "".join(cells[28]["source"])
    translation_code = "".join(cells[29]["source"])
    save_code = "".join(cells[30]["source"])
    final_report_markdown = "".join(cells[31]["source"])
    report_console_code = "".join(cells[32]["source"])
    report_save_code = "".join(cells[33]["source"])

    assert cells[27]["cell_type"] == "markdown"
    assert "## 6. Dataset Organization & Translation" in section_markdown

    assert cells[28]["cell_type"] == "code"
    assert "consolidated_rows = []" in consolidation_code
    assert "consolidated_df = pd.DataFrame(consolidated_rows)" in consolidation_code

    assert cells[29]["cell_type"] == "code"
    assert "sample_nutrients" in translation_code
    assert "translate_nutrients(" in translation_code

    assert cells[30]["cell_type"] == "code"
    assert "consolidated_dataset.csv" in save_code

    assert cells[31]["cell_type"] == "markdown"
    assert "## Final Report" in final_report_markdown

    assert cells[32]["cell_type"] == "code"
    assert "NUTRITION LABEL OCR - DATA COLLECTION REPORT" in report_console_code
    assert "--- DATASETS ---" in report_console_code
    assert "--- DOWNLOADS ---" in report_console_code
    assert "--- AUGMENTATION ---" in report_console_code
    assert "--- STORAGE ---" in report_console_code
    assert "--- OUTPUT FILES ---" in report_console_code

    assert cells[33]["cell_type"] == "code"
    assert "report_" in report_save_code
    assert "CONFIG['processed_dir']" in report_save_code
