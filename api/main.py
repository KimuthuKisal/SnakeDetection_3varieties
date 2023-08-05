from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

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
    #resize the image to 256 256 before predict
    predictions = MODEL.predict(img_batch)
    predicted_class = snake_labels[np.argmax(predictions[0])]
    # predicted_class = np.argmax(predictions[0])
    confidence = round(100 * (np.max(predictions[0])), 2)
    print('class' , predicted_class,'confidence' , float(confidence))
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run( app, host='localhost', port=8000 )