from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "localhost:8501/predict" #fix this

MODEL = tf.keras.models.load_model("../models/fourth_version")

snake_labels = ['Cobra', 'Sri Lankan Krait', 'Whip snakes' ]


@app.get("/ping")
async def  ping():
    return "Hello"

def read_file_as_image(data) -> np.ndarray:
    image = np.array( Image.open( BytesIO(data) ) )
    return image

@app.post("/predict")
async def predict( file: UploadFile = File(...) ):
    image = read_file_as_image( await file.read() )
    img_batch = np.expand_dims( image, 0 )
    #resize the image to 256 256 before predict - may be not essential
    predictions = MODEL.predict(img_batch)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = snake_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    print('class' , predicted_class,'confidence' , float(confidence))

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run( app, host='localhost', port=8000 )