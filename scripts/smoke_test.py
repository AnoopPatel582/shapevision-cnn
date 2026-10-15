"""Run every notebook code cell on a tiny generated dataset without saving files in the repo."""

import json
import os
import tempfile
from pathlib import Path

import matplotlib
import pandas as pd
import torch

from generate_demo_data import generate_demo_data


matplotlib.use("Agg")
torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / "ShapeVision_CNN.ipynb").read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
starting_directory = Path.cwd()

with tempfile.TemporaryDirectory() as temp_dir:
    try:
        os.chdir(temp_dir)
        generate_demo_data(Path("data"), train_per_class=4, test_count=2, size=32)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if index == 3:
                assert all(setting in source for setting in (
                    "IMAGE_SIZE = 224", "BATCH_SIZE = 64", "EPOCHS = 20"
                ))
                source = source.replace("IMAGE_SIZE = 224", "IMAGE_SIZE = 32")
                source = source.replace("BATCH_SIZE = 64", "BATCH_SIZE = 4")
                source = source.replace("EPOCHS = 20", "EPOCHS = 1")
            if index == 17:
                source = source.replace("plt.show()", "plt.close()")
            exec(compile(source, str(root / "ShapeVision_CNN.ipynb"), "exec"), namespace)

        result = pd.read_csv("artifacts/submission.csv")
        assert result.columns.tolist() == ["ID", "Label"]
        assert len(result) == 2
        assert set(result["Label"]).issubset({0, 1})
        print("PASS: all notebook code cells, checkpoint, and two predictions")
    finally:
        os.chdir(starting_directory)
