# 🌊 Water Segmentation Using Harmonized Landsat & Sentinel-2 Data

This project applies deep learning to perform **semantic segmentation** for **water detection** using harmonized **Landsat/Sentinel-2 satellite imagery**. It includes a full pipeline from **data preprocessing** to **Flask-based deployment**, enabling real-time segmentation via a web interface.

---

## 🚀 Live Demo

🔗 [Watch the demo](https://drive.google.com/file/d/12rTHnjuhvOT8kHy1g7x5sySOL-yGTKRP/view?usp=sharing)

---

## 🧠 Key Highlights

- ✅ **Multispectral Data Processing**: Efficiently handled Harmonized Landsat/Sentinel-2 imagery and selected key spectral bands for water detection.
- ✅ **Data Preprocessing**: Performed resizing, normalization, and augmentation tailored to satellite images.
- ✅ **Modeling**: Trained a deep learning segmentation model using U-Net-based architecture for pixel-wise classification.
- ✅ **Deployment**: Built a **Flask API** with HTML frontend to support image uploads and real-time segmentation.

---

## 🛠️ Tech Stack

| Component        | Tools/Frameworks              |
|------------------|-------------------------------|
| Language         | Python                        |
| Notebook         | Jupyter Notebook              |
| Deep Learning    | TensorFlow / Keras            |
| Web Framework    | Flask                         |
| Frontend         | HTML (Jinja templates)        |
| Data             | Harmonized Landsat / Sentinel-2 |

---

## 📁 Project Structure
#### ├── app.py # Flask web application
#### ├── templates/
##### │ ├── index.html # Home/upload page
##### │ └── result.html # Segmentation results display
#### ├── sentinel-2-water-segmentation.ipynb # Model training & data processing notebook
#### ├── static/ # Static files (if used)
##### │ └── (optional assets/images)
#### ├── model/ # Trained segmentation model
##### │ └── water_segmentation_model.h5
#### ├── requirements.txt # Python dependencies
##### └── README.md # Project documentation
## Getting Started
### 1. Clone the repository
```
git clone https://github.com/yourusername/water-segmentation-landsat-sentinel.git
cd water-segmentation-landsat-sentinel
```

### 2. Create a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt

```
### 4. Run the Flask app
```
python app.py
```
##📓 Model Training
Training and data preprocessing steps are documented in the Jupyter notebook:

sentinel-2-water-segmentation.ipynb

Includes:

Reading TIFF images

Selecting spectral bands

Creating masks for water

Model architecture and training

## 🌐 Web App Functionality
Upload multispectral or RGB-converted image

Run real-time segmentation via Flask backend

View and download the segmented output

## 📌 Notes
The model was trained on a custom dataset using bands like [B2, B3, B4, B5, B6, B7] from Sentinel-2 imagery.

Input image preprocessing must align with the model's expected format (e.g., resized to 128×128).

