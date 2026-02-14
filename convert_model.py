import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("dogbreed.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("dogbreed.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion complete. dogbreed.tflite created.")