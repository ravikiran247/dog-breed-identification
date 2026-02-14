import os
import numpy as np
import tensorflow as tf
import gdown
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

FILE_ID = "1uWGnAKEvL6jdzPd84nx2zJgjniNEXgmH"
MODEL_PATH = "dogbreed.h5"

model = None


def get_model():
    global model

    if model is None:

        # Download only if not present
        if not os.path.exists(MODEL_PATH):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

        model = tf.keras.models.load_model(MODEL_PATH)

    return model


class_names = [
    'affenpinscher','beagle','appenzeller','basset','bluetick',
    'boxer','cairn','doberman','german_shepherd','golden_retriever',
    'kelpie','komondor','leonberg','mexican_hairless','pug',
    'redbone','shih-tzu','toy_poodle','vizsla','whippet'
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    model = get_model()
    prediction = model.predict(img_array)

    confidence = float(np.max(prediction)) * 100
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(confidence, 2)

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=confidence,
        image_path=filepath
    )