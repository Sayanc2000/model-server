import uvicorn
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# from tensorflow.keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array

app = FastAPI()
model = load_model('model_88%.h5')


def process_image():
    img = load_img('test.png', target_size=(50,50))
    img = img_to_array(img)
    img = img.reshape(1, 50, 50, 3)
    img = img.astype('float32')

    return img


def read_image(file):
    strin = file.decode('utf-8')
    with open('test.png', 'w') as f:
        f.write(strin)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/image")
async def predict(file: UploadFile = File(...)):
    try:
        f = await file.read()
        print(file)
    except Exception as e:
        print(e)
        return {"response": "error"}
    try:
        read_image(f)
        img = process_image()
        x = model.predict(img).argmax()
        
    except Exception as e:
        print(e)
        return {"response": "error"}

    return {"response": x}

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")