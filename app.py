import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("dogbreed.h5")

# Class names (same order used during training)
class_names = [
    'affenpinscher','beagle','appenzeller','basset','bluetick',
    'boxer','cairn','doberman','german_shepherd','golden_retriever',
    'kelpie','komondor','leonberg','mexican_hairless','pug',
    'redbone','shih-tzu','toy_poodle','vizsla','whippet'
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template("result.html",
                               prediction=predicted_class,
                               image_path=filepath)

    return "No file uploaded"


if __name__ == "__main__":
    app.run()