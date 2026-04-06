import json
import re
from pathlib import Path

import src.utils as utils


def test_utils_public_api_exports_augmentation_and_translation_helpers():
    assert "apply_augmentation" in utils.__all__
    assert "translate_nutrients" in utils.__all__
    assert callable(utils.apply_augmentation)
    assert callable(utils.translate_nutrients)


def test_section6_uses_detected_image_column_in_match_original_row():
    notebook_path = Path("notebooks/01_data_collection.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "image_columns.get(dataset_name, ['image_url'])" in code
    assert re.search(
        r"match_original_row\(\s*original_df,\s*row\.get\('url'\),\s*image_url_column=",
        code,
    )
