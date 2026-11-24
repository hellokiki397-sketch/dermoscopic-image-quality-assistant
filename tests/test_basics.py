import numpy as np
from app import assess_image, preprocess

def test_functions_return_expected_shapes():
    image = np.full((300, 400, 3), 120, dtype=np.uint8)
    metrics = assess_image(image)
    assert metrics["Width (px)"] == 400
    assert metrics["Height (px)"] == 300
    assert 0 <= metrics["Heuristic quality score (/100)"] <= 100
    result = preprocess(image, remove_dark_lines=False, apply_clahe=False)
    assert result.shape == image.shape
