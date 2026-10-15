"""Offline checks of the notebook and its generated demonstration data."""

import ast
import json
import runpy
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "ShapeVision_CNN.ipynb"


def notebook_cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def class_namespace():
    namespace = {"nn": nn, "torch": torch, "Dataset": Dataset, "Image": Image}
    for cell in notebook_cells():
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        definitions = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if definitions:
            module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
            exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


class NotebookChecks(unittest.TestCase):
    def test_all_code_cells_parse(self):
        for index, cell in enumerate(notebook_cells()):
            if cell["cell_type"] == "code":
                with self.subTest(cell=index):
                    ast.parse("".join(cell["source"]))

    def test_residual_cnn_forward_and_backward(self):
        torch.set_num_threads(2)
        model = class_namespace()["CNN"]()
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 1))
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([[0.0], [1.0]]))
        loss.backward()
        gradient = model.stem[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all().item())

    def test_demo_images_load_through_notebook_transforms(self):
        generator = runpy.run_path(str(ROOT / "scripts" / "generate_demo_data.py"))["generate_demo_data"]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "demo"
            generator(output, train_per_class=4, test_count=2, size=32)
            self.assertEqual(len(list((output / "train" / "train" / "0").glob("*.png"))), 4)
            self.assertEqual(len(list((output / "train" / "train" / "1").glob("*.png"))), 4)
            self.assertEqual(len(list((output / "test" / "test").glob("*.png"))), 2)
            with self.assertRaises(FileExistsError):
                generator(output)

            transforms = {"v2": v2, "torch": torch, "IMAGE_SIZE": 32}
            exec("".join(notebook_cells()[8]["source"]), transforms)
            path = output / "train" / "train" / "0" / "image_000.png"
            dataset = class_namespace()["TrainDataset"]([str(path)], [0], transforms["train_tfms"])
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(label, 0)
            self.assertTrue(torch.isfinite(image).all().item())


if __name__ == "__main__":
    unittest.main()
