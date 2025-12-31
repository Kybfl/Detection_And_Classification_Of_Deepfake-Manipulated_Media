# 🕵️‍♂️ Deepfake Detection and Classification System

**A robust, modular, and audit-ready deep learning application for detecting and classifying manipulated media content.**

## 📖 Overview

This project aims to detect and classify deepfake videos and images using state-of-the-art computer vision and deep learning techniques. Unlike standard binary classifiers (Real vs. Fake), this system identifies the specific manipulation technique used.

It features a hybrid architecture combining **OpenCV DNN** for robust face detection and **EfficientNet-B0** for feature extraction and classification. The system is wrapped in a user-friendly **Streamlit** web interface and includes a secure **Audit Logging** system for legal non-repudiation.

## ✨ Key Features

* **Multi-Class Classification:** Detects **Original** content and 5 types of Deepfakes:
    * Deepfakes
    * Face2Face
    * FaceSwap
    * FaceShifter
    * NeuralTextures
* **Hybrid AI Architecture:**
    * **Face Detection:** OpenCV DNN (ResNet-10 SSD) for high accuracy in various lighting conditions.
    * **Classification:** EfficientNet-B0 (optimized via Transfer Learning).
* **Secure Audit Logging:** Every analysis is hashed (SHA-256) and logged into a local SQLite database to ensure data integrity and traceability.
* **Real-Time Analysis:** Supports both static image upload and frame-by-frame video analysis with dynamic confidence score visualization.
* **User-Friendly GUI:** Interactive web interface built with Streamlit.

---

## 📸 Screenshots

### 1. Home Dashboard & System Status
![Home Page](D:\Git_Projeler\Detection_And_Classification_Of_Deepfake-Manipulated_Media\App_ScreenShots\homepage.png)
*Displays system readiness, model status, and supported manipulation types.*

### 2. Image Analysis Module
![Image Analysis](D:\Git_Projeler\Detection_And_Classification_Of_Deepfake-Manipulated_Media\App_ScreenShots\pictureanalyse.png)
*Detects faces, applies classification, and provides a probability distribution chart.*

### 3. Real-Time Video Analysis
![Video Analysis](D:\Git_Projeler\Detection_And_Classification_Of_Deepfake-Manipulated_Media\App_ScreenShots\videoanalyse.png)
*Frame-by-frame analysis with dynamic confidence tracking.*

---

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV (cv2)
* **Interface:** Streamlit
* **Data Processing:** NumPy, Pillow
* **Database:** SQLite

---

## 🚀 Installation

Follow these steps to set up the project locally.



---

## 📊 Dataset Preparation (Optional)

If you do not already have the FaceForensics++ dataset installed, you can download and preprocess it using the helper scripts provided in the project.

Navigate to the following directory:

**Dataset_Operations_Helper**

Open the following files (or their corresponding Python scripts):

* **Dataset_Downloader**
* **Video_Frame_Extracter**

Update all file and directory paths inside these scripts so that they match your local storage location.

Run the following commands in your terminal to download the video data and extract frames:

```bash
# Download video data
python download.py

# Extract frames from compressed videos
python extract_compressed_videos.py
```

After completing these steps, the dataset will be ready for training or inference.

---

## ⚙️ Model Configuration

Before running the application, you must specify the path to your trained deepfake detection model.

1. Open the **app.py** file in your preferred code editor.
2. Locate the variable named **MODEL_PATH**.
3. Update this variable with the absolute or relative path to your trained .keras model file.

**Example:**

```python
# Example in app.py
MODEL_PATH = "path/to/your/model/best_deepfake_model.keras"
```

Ensure that the model file exists at the specified location and is accessible.

---

## 🎯 Running the Application

Once the dataset preparation (if required) and model configuration are complete, you can launch the application interface.

1. Open a terminal or command prompt.
2. Navigate to the root directory of the project.
3. Run the following command:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser, typically at:

```
http://localhost:8501
```

You can now upload your data and test the trained deepfake detection model through the web interface.

---

## 📝 Notes

* Make sure all required Python dependencies are installed before running the application.
* Dataset preparation is optional if you already have a preprocessed dataset or only intend to perform inference using a trained model.
* For large datasets, ensure sufficient disk space and processing time.

---

## Project File Structure

# 🕵️‍♂️ Deepfake Detection and Classification System

**A robust, modular, and audit-ready deep learning application for detecting and classifying manipulated media content.**

![Project Banner](placeholder_for_banner.png)
*(Please insert your project banner image here)*

## 📖 Overview

This project is a Computer Engineering Capstone Project designed to detect and classify deepfake videos and images. It utilizes a hybrid architecture combining **OpenCV DNN** (for face detection) and **EfficientNet-B0** (for classification).

Unlike standard binary classifiers (Real vs. Fake), this system identifies the specific manipulation technique used:
* **Original**
* **Deepfakes**
* **Face2Face**
* **FaceSwap**
* **FaceShifter**
* **NeuralTextures**

The system features a **Streamlit** web interface for real-time analysis and includes a secure **Audit Logging** system (SHA-256 & SQLite) to ensure data integrity.

---

## 📂 Project Structure

For the application to run without errors, ensure your local directory structure matches the hierarchy below:

```
Detection_And_Classification_Of_Deepfake-Manipulated_Media/
│
├── README.md                          # Project documentation and guide
│
├── App/                               # Main application directory
│   ├── app.py                         # Streamlit web interface & main entry point
│   ├── db_manager.py                  # SQLite database management & audit logging
│   ├── model_handler.py               # Model loading & inference logic
│   │
│   └── DNN_Files/                     # OpenCV DNN model files for face detection
│       ├── deploy.prototxt            # DNN architecture definition
│       └── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained face detection model
│
├── App_ScreenShots/                   # Application screenshots & demos
│   ├── homepage.png                   # Home dashboard screenshot
│   ├── pictureanalyse.png             # Image analysis interface screenshot
│   └── videoanalyse.png               # Video analysis interface screenshot
│
├── Dataset_Operations_Helper/         # Dataset preparation utilities
│   ├── Dataset_Downloader.txt         # Instructions for downloading FaceForensics++
│   ├── download.py                    # Script to download FaceForensics++ dataset
│   ├── Video_Frame_Extracter.txt      # Instructions for video frame extraction
│   └── extract_compressed_videos.py   # Script to extract frames from compressed videos
│
├── Detection_Model/                   # Trained deepfake detection model
│   ├── best_deepfake_model.keras      # Trained EfficientNet-B0 model weights
│   └── training_results.png           # Model training performance visualization
│
└── .git/                              # Git repository metadata
```

### Directory Descriptions:

- **App/**: Contains the main application code, including the Streamlit interface, database manager, and model inference handler.
- **App/DNN_Files/**: OpenCV DNN pre-trained models for face detection using ResNet-10 SSD architecture.
- **App_ScreenShots/**: Visual screenshots of the web interface for documentation and demonstration.
- **Dataset_Operations_Helper/**: Helper scripts and instructions for downloading and preprocessing the FaceForensics++ dataset.
- **Detection_Model/**: The trained deepfake classification model (EfficientNet-B0) and its training results.

---

## 📜 Licenses

This project includes components from multiple sources:

- **Streamlit**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Pillow**

---

## 📚 FaceForensics++ Citation

The detection model in this project was trained using the **FaceForensics++** dataset. If you use this work in academic research, please cite the original FaceForensics++ paper:

```bibtex
@inproceedings{rössler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={Rössler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nießner, Matthias},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

**Dataset Link:** [FaceForensics++ - Technical University of Munich](https://github.com/ondyari/FaceForensics)

**Citation Details:**
- **Authors:** Rössler et al.
- **Year:** 2019
- **Conference:** IEEE/CVF International Conference on Computer Vision (ICCV)
- **DOI:** 10.1109/ICCV.2019.00928

The FaceForensics++ dataset is a large-scale forensics dataset containing pristine videos and manipulated versions created with five state-of-the-art face manipulation techniques, including DeepFakes, Face2Face, FaceSwap, FaceShifter, and Neural Textures - all manipulation methods supported by this detection system.

