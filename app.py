import os
import numpy as np
import rasterio
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"tif", "tiff"}

MODEL_PATH = os.path.join(os.getcwd(), "models", "best_model.keras")

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

@tf.keras.utils.register_keras_serializable()
def iou(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'dice_coefficient': dice_coefficient, 'iou': iou}
)

SELECTED_BANDS = [3, 4, 5]  
def preprocess_image(image_path):
    with rasterio.open(image_path) as src:
        image = np.stack([src.read(b) for b in SELECTED_BANDS], axis=-1)  

    resized_bands = [cv2.resize(image[:, :, i], (128, 128), interpolation=cv2.INTER_CUBIC) for i in range(image.shape[-1])]
    image = np.stack(resized_bands, axis=-1)

    image = image.astype(np.float32) / 255.0  

    return image

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_tiff_to_png(tiff_path, png_path):
    with rasterio.open(tiff_path) as src:
        image = np.stack([src.read(b) for b in SELECTED_BANDS], axis=-1)
    
    image = (image / np.max(image) * 255).astype(np.uint8)
    
    Image.fromarray(image).save(png_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_and_predict():
    file = request.files['file']
    
    if file and file.filename.endswith('.tif'):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        png_filename = filename.replace('.tif', '.png')
        convert_tiff_to_png(filename, png_filename)

        image = preprocess_image(filename)
        image = np.expand_dims(image, axis=0) 

        prediction = model.predict(image)[0]  

        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], "predicted_mask.png")
        plt.imsave(mask_path, prediction.squeeze(), cmap='gray')

        print("Generated Files:", png_filename, mask_path)
        
        return render_template("result.html", filename=os.path.basename(png_filename), mask_filename="predicted_mask.png")
    
    return "Invalid file format!", 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
