import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import os

# Dataset path (20 breeds)
train_path = "dataset_20/train"

# Image size
IMAGE_SIZE = [128, 128]

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=6,
    class_mode='categorical'
)

# Load VGG19 model
base_model = VGG19(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False
)

# Freeze VGG layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
from tensorflow.keras.layers import GlobalAveragePooling2D

x = GlobalAveragePooling2D()(base_model.output)
prediction = Dense(20, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=prediction)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_generator,
    epochs=6
)

# Save model
model.save("dogbreed.h5")

print("Model Training Completed and Saved Successfully!")