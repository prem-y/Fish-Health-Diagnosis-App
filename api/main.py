from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load the model without custom objects
MODEL = tf.keras.models.load_model("../SavedModel/Fish_Disease_Detection")


CLASS_NAMES = ["Healthy Fish", "Infected Fish"]
INPUT_SIZE = (224, 224)


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    if image.size != INPUT_SIZE:
        image = image.resize(INPUT_SIZE).convert('RGB')
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
