from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request,
    Form,
)


from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from PIL import Image
import io

from app.model import predict_image
from app.schemas import PredictionResponse
from app.agent import ask_food_agent


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Food Classifier API",
    description="FastAPI food image classifier using EfficientNet-B0",
    version="1.0.0",
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory="templates"
)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


# =========================================================
# HOME
# =========================================================

@app.get(
    "/",
    include_in_schema=False,
)
async def get(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model": "EfficientNet-B0",
        "classes": [
            "pizza",
            "steak",
            "sushi",
        ],
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    question: str = Form(""),
):

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:

        content = await file.read()

        image = Image.open(
            io.BytesIO(content)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image.",
        )

    # -----------------------------------------------------
    # ML prediction
    # -----------------------------------------------------

    result = predict_image(
        image=image
    )

    # -----------------------------------------------------
    # User question or default overview
    # -----------------------------------------------------

    if not question.strip():

        question = (
            "Give me a short overview of this food. "
            "Include what it is, common ingredients or preparation, "
            "basic nutritional information, and approximate calories."
        )

    # -----------------------------------------------------
    # AI agent
    # -----------------------------------------------------

    agent_response = ask_food_agent(
        prediction=result["prediction"],
        confidence=result["confidence"],
        question=question,
    )

    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "agent_response": agent_response,
    }
