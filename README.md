# Food Classifier API

A FastAPI-based web application that serves a PyTorch EfficientNet-B0 image classification model for recognizing **pizza, steak, and sushi**.

The application provides a browser-based interface where users can upload a food image and receive a prediction, confidence score, and class probabilities.

## Features

* Food image classification using EfficientNet-B0
* Classes:

  * Pizza
  * Steak
  * Sushi
* FastAPI REST API
* Browser-based image upload interface
* Image preview before prediction
* Confidence score
* Probability distribution for each class
* Basic image validation
* Invalid/corrupted image handling
* Pydantic response validation
* Interactive Swagger API documentation

## Project Architecture

```text
                         ┌──────────────────┐
                         │      Browser     │
                         │   HTML + CSS +   │
                         │    JavaScript    │
                         └────────┬─────────┘
                                  │
                              POST /predict
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         │   main.py        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Image Processing│
                         │      PIL         │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  EfficientNet-B0 │
                         │    PyTorch       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Prediction +     │
                         │ Confidence +     │
                         │ Probabilities    │
                         └──────────────────┘
```

## Project Structure

```text
food-classifier-api/
│
├── app/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
│
├── model/
│   └── food_classifier.pth
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── test_model.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

> The trained `.pth` model file is excluded from Git using `.gitignore`.

## Model

The classifier uses **EfficientNet-B0** from Torchvision.

The pretrained EfficientNet-B0 feature extractor was used with a custom classification head for the three food categories.

Input images are resized to:

```text
224 × 224
```

The images are normalized using the ImageNet normalization values:

```text
Mean:
[0.485, 0.456, 0.406]

Standard deviation:
[0.229, 0.224, 0.225]
```

The model returns probabilities using Softmax.

## API

### Home

```http
GET /
```

Returns the web interface.

### Prediction

```http
POST /predict
```

Accepts an image file and returns the classification result.

Example response:

```json
{
    "filename": "pizza.jpg",
    "prediction": "pizza",
    "confidence": 0.8812,
    "probabilities": {
        "pizza": 0.8812,
        "steak": 0.0715,
        "sushi": 0.0473
    }
}
```

## Swagger Documentation

FastAPI automatically provides interactive API documentation.

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

You can upload an image directly from Swagger and test the `/predict` endpoint.

## Installation

Clone the repository:

```bash
git clone https://github.com/LOKESH1027-V/food-classifier-api.git
```

Enter the project:

```bash
cd food-classifier-api
```

Install dependencies using `uv`:

```bash
uv sync
```

## Running the Application

Start the FastAPI development server:

```bash
uv run uvicorn app.main:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000/
```

## Testing the Model

The project includes a simple test script:

```bash
uv run python test_model.py
```

This script loads test images and displays the model's predictions and confidence scores.

## Important Note

The current model was trained specifically for three food categories:

```text
pizza
steak
sushi
```

Images outside these categories may still receive a prediction because the classifier has no dedicated `other` class.

A confidence threshold is currently used to classify sufficiently low-confidence predictions as `unknown`.

A future version can improve this behavior by training the model with an additional `other` class.

## Technologies

* Python
* PyTorch
* Torchvision
* EfficientNet-B0
* FastAPI
* Pydantic
* Uvicorn
* Jinja2
* HTML
* CSS
* JavaScript
* uv

## Future Improvements

* Add a dedicated `other` class
* Improve unknown-image detection
* Add authentication
* Add prediction history
* Store uploaded images
* Improve frontend design
* Deploy the API
* Integrate the classifier with an AI agent
