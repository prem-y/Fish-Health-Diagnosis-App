from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import imghdr

app = FastAPI()

MODEL = tf.keras.models.load_model("../SavedModel/Fish_Disease_Detection")  #load model
CLASS_NAMES = ["Healthy Fish", "Infected Fish"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive!!"  #check if system is running


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_type = imghdr.what("", h=file.file.read(1024))
    if not file_type or file_type not in {"jpeg", "png", "gif"}:
        raise HTTPException(status_code=400, detail="Invalid file.check the filetype and try again")   #input validation

    image = read_file_as_image(await file.read())

    # Make prediction
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence) #return the result
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
