import os
from pathlib import Path

test_path = Path("Source")
img = str(test_path / "france.tif")
print(img.split("\\")[1])
