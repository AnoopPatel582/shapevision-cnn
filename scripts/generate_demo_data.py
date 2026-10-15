"""Generate a small, local circle-versus-square dataset for the notebook."""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def draw_example(label: int, size: int, rng: random.Random) -> Image.Image:
    background = tuple(rng.randint(225, 255) for _ in range(3))
    image = Image.new("RGB", (size, size), background)
    canvas = ImageDraw.Draw(image)
    diameter = rng.randint(size // 2, size * 3 // 4)
    left = rng.randint(2, size - diameter - 2)
    top = rng.randint(2, size - diameter - 2)
    box = (left, top, left + diameter, top + diameter)
    if label == 0:
        canvas.ellipse(box, fill=(rng.randint(150, 230), 35, 35))
    else:
        canvas.rectangle(box, fill=(35, 75, rng.randint(150, 230)))
    return image


def generate_demo_data(output: Path, train_per_class: int = 20, test_count: int = 8,
                       size: int = 96, seed: int = 42) -> None:
    if train_per_class < 2 or test_count < 1 or size < 24:
        raise ValueError("Use at least two training images per class, one test image, and size 24 or greater")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new output folder")

    rng = random.Random(seed)
    for label in (0, 1):
        folder = output / "train" / "train" / str(label)
        folder.mkdir(parents=True)
        for index in range(train_per_class):
            draw_example(label, size, rng).save(folder / f"image_{index:03d}.png")

    test_folder = output / "test" / "test"
    test_folder.mkdir(parents=True)
    for index in range(test_count):
        label = index % 2
        draw_example(label, size, rng).save(test_folder / f"image_{index:03d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()
    generate_demo_data(args.output, args.train_per_class, args.test_count, args.size)
    print(f"Demo images created in {args.output}")
