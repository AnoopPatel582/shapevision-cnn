# ShapeVision CNN

ShapeVision CNN is a PyTorch notebook for binary image classification. It uses a four-block residual CNN, image augmentation, stratified validation, and a saved best-model checkpoint. You can run it with your own two-class image dataset or generate a small circle-versus-square demo dataset locally.

## What this adaptation adds

- A reproducible synthetic demo so the project can be tried without the original dataset.
- CPU-safe device reporting, consistent training timers, and deterministic file ordering.
- Clear dataset validation, flattened prediction arrays, and a best-F1 checkpoint used for test predictions.
- Offline tests for notebook syntax, data transforms, and a model training step.

## Set up

Use Python 3.11 or newer. From this repository's directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a quick demo, generate images in the expected folder structure:

```powershell
.\.venv\Scripts\python.exe scripts\generate_demo_data.py
```

The generator creates simple **class 0 circles** and **class 1 squares**. These are only demonstration images; they are not the original training dataset. It refuses to overwrite an existing `data` folder.

## Use your own data

The notebook expects:

```text
data/
├── train/train/
│   ├── 0/          # first class
│   └── 1/          # second class
└── test/test/      # unlabeled images
```

Supported image formats are PNG, JPG, JPEG, and WebP. Each training class needs at least two images. The notebook reserves at least one image from each class for stratified validation. Change `TRAIN_DIR` and `TEST_DIR` in its configuration cell if your files are elsewhere. Large or private datasets stay outside Git.

## Train and predict

```powershell
.\.venv\Scripts\python.exe -m notebook ShapeVision_CNN.ipynb
```

Run the notebook cells in order. By default, training uses 224-pixel images, batch size 64, 20 epochs, Adam, BCE-with-logits loss, and cosine learning-rate scheduling. The notebook chooses CUDA when available and otherwise runs on CPU. Full training can be slow on a CPU.

Each epoch reports training loss, validation accuracy, validation F1, learning rate, and elapsed time. The checkpoint with the highest validation F1 is saved to `artifacts/best_model.pt`. Test predictions use that checkpoint and are written to `artifacts/submission.csv` with `ID` and `Label` columns.

## Verify the code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use temporary generated images and do not need a downloaded dataset. They check syntax, transforms, output shape, and a backward pass. A reduced one-epoch run on generated images has been used as a runtime check; it does not establish accuracy on a real dataset. No original model weights or measured real-data performance are included.

To repeat the reduced end-to-end notebook run without leaving data or checkpoints in the repository:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```
