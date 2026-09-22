# Food Classifier API

An AI-powered food image classification application that combines a deep learning image classifier with a Groq-powered AI food assistant.

The application identifies food images as pizza, steak, or sushi using an EfficientNet-B0 model and then uses an AI agent to answer questions about the detected food.

## Features

* Food image classification using EfficientNet-B0
* Transfer learning with PyTorch and Torchvision
* Supports pizza, steak, and sushi
* Confidence-based prediction
* Returns `unknown` for low-confidence predictions
* Class probability scores
* FastAPI REST API
* Browser-based frontend
* Optional user questions about the detected food
* AI food assistant powered by Groq
* LangChain tool calling
* Food information and nutrition tools
* Automatic API documentation with Swagger UI
* Git LFS for the trained model
* Ready for cloud deployment

## Architecture

```text
                    User
                     |
                     v
              Browser Frontend
                     |
              Image + Question
                     |
                     v
              FastAPI /predict
                     |
                     v
             EfficientNet-B0
                     |
          +----------+----------+
          |                     |
     Prediction            Probabilities
          |                     |
          +----------+----------+
                     |
                     v
                Food Agent
                     |
               Groq LLM
                     |
          +----------+----------+
          |                     |
   Food Information       Nutrition Tool
          |                     |
          +----------+----------+
                     |
                     v
              Final AI Response
                     |
                     v
               Browser UI
```

## Tech Stack

### Machine Learning

* Python
* PyTorch
* Torchvision
* EfficientNet-B0
* Transfer Learning
* Pillow

### Backend

* FastAPI
* Uvicorn
* Pydantic
* Python Multipart

### AI Agent

* LangChain
* LangChain Groq
* Groq LLM
* Tool Calling

### Frontend

* HTML
* CSS
* JavaScript
* Jinja2 Templates

### Deployment

* GitHub
* Git LFS
* Render

## Project Structure

```text
food-classifier-api/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   └── agent.py
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
├── test_agent.py
├── test_tool.py
├── test_tool_agent.py
│
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

## Machine Learning Model

The application uses an EfficientNet-B0 model with transfer learning.

The pretrained feature extractor is frozen and the classifier is replaced with a custom three-class classification layer.

```text
EfficientNet-B0
      |
      v
Feature Extraction
      |
      v
Custom Classifier
      |
      +---- Pizza
      +---- Steak
      +---- Sushi
```

The model accepts images resized to:

```text
224 x 224
```

The images are normalized using the standard ImageNet mean and standard deviation.

The trained model is stored as:

```text
model/food_classifier.pth
```

The model file is managed using Git LFS.

## Prediction

The API returns:

* Predicted food
* Confidence score
* Probability for each supported class

Example:

```json
{
    "prediction": "sushi",
    "confidence": 0.8462,
    "probabilities": {
        "pizza": 0.0269,
        "steak": 0.1269,
        "sushi": 0.8462
    }
}
```

A confidence threshold is used to reduce unreliable predictions. Images below the configured threshold are returned as:

```text
unknown
```

## AI Food Assistant

After classification, the detected food and the user's optional question are passed to an AI food assistant.

For example, a user can upload an image and ask:

```text
How many calories does this food have?
```

The agent determines whether it needs additional information and can use specialized tools.

### Available Tools

#### Food Information Tool

Provides basic information about:

* Ingredients
* Preparation
* General description

#### Nutrition Tool

Provides approximate nutritional information such as calories.

The agent then generates a concise natural-language response.

## API Endpoints

### Home

```http
GET /
```

Returns the browser-based application.

### Health Check

```http
GET /health
```

Returns the API health status and supported classes.

### Prediction

```http
POST /predict
```

Accepts:

* `file` — food image
* `question` — optional question about the detected food

Example response:

```json
{
    "filename": "food.jpg",
    "prediction": "pizza",
    "confidence": 0.91,
    "probabilities": {
        "pizza": 0.91,
        "steak": 0.06,
        "sushi": 0.03
    },
    "agent_response": "Pizza is a popular Italian dish..."
}
```

## Swagger Documentation

After starting the application, interactive API documentation is available at:

```text
http://localhost:8000/docs
```

Alternative documentation:

```text
http://localhost:8000/redoc
```

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/LOKESH1027-V/food-classifier-api.git
cd food-classifier-api
```

### 2. Install dependencies

This project uses `uv`.

```bash
uv sync
```

### 3. Configure the Groq API key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit the `.env` file.

### 4. Start the application

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## Environment Variables

The application requires:

```text
GROQ_API_KEY
```

For local development, store it in `.env`.

For production deployment, configure it as a secure environment variable in the hosting platform.

## Deployment

The application can be deployed as a FastAPI web service.

### Build Command

```bash
uv sync --frozen && uv cache prune --ci
```

### Start Command

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The `GROQ_API_KEY` must be configured as a production environment variable.

## Git LFS

The trained PyTorch model is stored using Git LFS because of its size.

Check tracked LFS files with:

```bash
git lfs ls-files
```

The model should appear as:

```text
model/food_classifier.pth
```

## Testing

The project contains separate tests for the model and AI agent components.

Examples:

```bash
uv run python test_model.py
```

```bash
uv run python test_agent.py
```

```bash
uv run python test_tool.py
```

## Design Decisions

### Why EfficientNet-B0?

EfficientNet-B0 provides a good balance between model accuracy and computational requirements, making it suitable for a lightweight image classification API.

### Why FastAPI?

FastAPI provides:

* Fast request handling
* Automatic API documentation
* Pydantic validation
* Simple file upload handling
* Easy integration with machine learning models

### Why an AI Agent?

The classifier determines what food is present in the image, while the AI agent provides additional natural-language information.

This separates the responsibilities:

```text
Computer Vision
      |
      v
"What food is this?"
      |
      v
AI Agent
      |
      v
"What would the user like to know about it?"
```

## Future Improvements

* Add more food classes
* Improve unknown-food detection
* Add an `other` class to the training dataset
* Add richer nutritional information
* Add food recommendation capabilities
* Add database support for prediction history
* Add authentication
* Add automated tests
* Improve model monitoring
* Add production logging
* Optimize inference performance

## Author

Lokesh P

GitHub:

https://github.com/LOKESH1027-V

## License

This project is intended for educational and portfolio purposes.
