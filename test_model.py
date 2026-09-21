from pathlib import Path

from PIL import Image

from app.model import predict_image


test_dir = Path("test_image")


for category_dir in test_dir.iterdir():

    if not category_dir.is_dir():
        continue

    print(f"\n===== {category_dir.name.upper()} =====")

    for image_path in category_dir.iterdir():

        try:
            image = Image.open(image_path).convert("RGB")

            result = predict_image(image)

            print(
                f"{image_path.name:20}"
                f" → {result['prediction']:8}"
                f" confidence={result['confidence']:.4f}"
            )

        except Exception as e:
            print(f"{image_path.name} → ERROR: {e}")