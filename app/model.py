import torch
import torchvision
from torchvision import transforms
from pathlib import Path


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "food_classifier.pth"


# =========================================================
# LOAD MODEL
# =========================================================

model = torchvision.models.efficientnet_b0(
    weights=None
)


model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(
        p=0.2,
        inplace=True
    ),
    torch.nn.Linear(
        in_features=1280,
        out_features=3,
        bias=True
    )
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)


model.eval()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# PREDICTION
# =========================================================

def predict_image(image):

    image = transform(image)

    image = image.unsqueeze(0)


    with torch.no_grad():

        output = model(image)


    probabilities = torch.softmax(
        output,
        dim=1
    )


    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


    confidence = probabilities[0][predicted_class].item()


    class_names = [
        "pizza",
        "steak",
        "sushi"
    ]


    threshold = 0.70


    if confidence < threshold:

        prediction = "unknown"

    else:

        prediction = class_names[predicted_class]


    class_probabilities = {

        class_names[i]: round(
            probabilities[0][i].item(),
            4
        )

        for i in range(len(class_names))

    }


    return {

        "prediction": prediction,

        "confidence": round(
            confidence,
            4
        ),

        "probabilities": class_probabilities

    }