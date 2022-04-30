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


def process_image(file):
    print('yes here')
    # img = load_img()
    img = img_to_array(file)
    img = img.reshape(1, 50, 50, 3)
    img = img.astype('float32')

    return img


def read_image(file) -> Image.Image:
    print('read_image')
    image = Image.open(BytesIO(file))
    return image


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/image")
async def predict(file: UploadFile = File(...)):
    try:
        f = await file.read()
        print(f, type(f))
    except Exception as e:
        print(e)
        return {"response": "error"}
    try:
        image = read_image(f)
        img = process_image(image)
        x = model.predict(img).argmax()
        
    except Exception as e:
        print(e)
        return {"response": "error"}

    return {"response": x}

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")