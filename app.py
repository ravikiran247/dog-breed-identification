import os
import numpy as np
import gdown
from flask import Flask, render_template, request
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

FILE_ID = "1uWGnAKEvL6jdzPd84nx2zJgjniNEXgmH"
MODEL_PATH = "dogbreed.tflite"

interpreter = None
input_details = None
output_details = None


def get_model():
    global interpreter, input_details, output_details

    if interpreter is None:

        # Download model if not present
        if not os.path.exists(MODEL_PATH):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    return interpreter


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

    img = Image.open(filepath).resize((128, 128))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    interpreter = get_model()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_names[np.argmax(prediction)]

    return render_template(
        "result.html",
        prediction=predicted_class,
        image_path=filepath
    )