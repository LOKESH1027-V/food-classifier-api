from fastapi import FastAPI,UploadFile,File,HTTPException
from PIL import Image
import io
from app.model import predict_image
from app.schemas import PredictionResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles



app=FastAPI()

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")
#-------------------------------------------------------------------
#Home
#-------------------------------------------------------------------
@app.get("/",include_in_schema=False)
async def get(request:Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


#-------------------------------------------------------------------
#Predict
#-------------------------------------------------------------------

@app.post("/predict",response_model=PredictionResponse)
async def predic(file: UploadFile=File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        ) 
    try:

        #Read the content
        content=await file.read()

        #Conver bytes to PIL
        image=Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or currupted image."
        )
    #Apply transform
    result=predict_image(image=image)


    return {
        "filename": file.filename,
        **result
    } 

