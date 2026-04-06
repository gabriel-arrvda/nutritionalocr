import json
from pathlib import Path


def test_section_3_has_operational_execution_evidence_workflow():
    notebook_path = Path("notebooks/01_data_collection.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    section3_markdown = None
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if "## 3. Data Exploration" in source:
            section3_markdown = source
            break

    assert section3_markdown is not None
    assert "notebooks/01_data_collection_test.ipynb" in section3_markdown
    assert "jupyter nbconvert --to notebook --execute notebooks/01_data_collection.ipynb" in section3_markdown
    assert "--output 01_data_collection_test.ipynb --output-dir notebooks" in section3_markdown
    assert "execution_count" in section3_markdown
    assert "11-17" in section3_markdown
    assert "output_type == 'error'" in section3_markdown
