import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os

class DeepfakeModel:
    def __init__(self, model_path):
        """
        Initialize Model and OpenCV DNN Face Detection system.
        """
        self.model_path = model_path
        self.class_names = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'Original']
        self.model = self._load_model()
        
        # Setup path for Face Detection models
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dnn_folder = os.path.join(base_dir, "DNN_Files") 
        
        prototxt_path = os.path.join(dnn_folder, "deploy.prototxt")
        model_weights_path = os.path.join(dnn_folder, "res10_300x300_ssd_iter_140000.caffemodel")
        
        self.face_net = None
        
        # Load the Face Detection Model
        if os.path.exists(prototxt_path) and os.path.exists(model_weights_path):
            try:
                print("Loading face detection model...")
                self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_weights_path)
                print("Face detection model ready!")
            except Exception as e:
                print(f"Error loading face detection model: {e}")
        else:
            print(f"Error: DNN files not found in {dnn_folder}")

    def _load_model(self):
        """
        Loads the Keras model. Returns None if an error occurs.
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print("Loading Deepfake model...")
            model = load_model(self.model_path)
            print("Deepfake model loaded successfully!")
            return model
        except Exception as e:
            print(f"Model loading error: {e}")
            return None

    def predict(self, frame):
        """
        Takes an image frame, detects the face using OpenCV DNN, and makes a prediction.
        """
        if self.model is None:
            return frame, None, "Model Could Not Be Loaded!"
        
        if self.face_net is None:
            return frame, None, "Face Detection Model Missing!"

        display_frame = frame.copy()
        (h, w) = frame.shape[:2]
        
        # Prepare image for DNN Face Detector
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        probs = None
        prediction_text = "Face Not Detected" # Default

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract confidence
            confidence_score = detections[0, 0, i, 2]

            # Filter out weak detections (Threshold: 50%)
            if confidence_score > 0.5:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding boxes fall within the dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Compute width and height
                w_box = endX - startX
                h_box = endY - startY
                
                # Ignore small boxes (noise)
                if w_box < 20 or h_box < 20:
                    continue

                # --- TRAINING CROPPING LOGIC (20% Margin) ---
                margin = 0.20
                x_new = max(0, int(startX - w_box * margin))
                y_new = max(0, int(startY - h_box * margin))
                w_new = int(w_box * (1 + 2 * margin))
                h_new = int(h_box * (1 + 2 * margin))

                # Boundary check
                if x_new + w_new > w: w_new = w - x_new
                if y_new + h_new > h: h_new = h - y_new

                face_img = frame[y_new:y_new+h_new, x_new:x_new+w_new]
                
                # Empty image check
                if face_img.size == 0: continue

                try:
                    # EfficientNet Preprocessing
                    face_resized = cv2.resize(face_img, (224, 224))
                    img_array = img_to_array(face_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    # Prediction
                    predictions = self.model.predict(img_array, verbose=0)
                    probs = predictions[0]
                    
                    # Parse Results
                    class_idx = np.argmax(probs)
                    confidence = probs[class_idx] * 100
                    class_name = self.class_names[class_idx]
                    
                    # Drawing Operations
                    color = (0, 255, 0) if class_name == 'Original' else (0, 0, 255)
                    label = f"{class_name}: %{confidence:.1f}"
                    
                    cv2.rectangle(display_frame, (x_new, y_new), (x_new+w_new, y_new+h_new), color, 2)
                    cv2.putText(display_frame, label, (x_new, y_new-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    prediction_text = f"**{class_name}** (%{confidence:.2f})"
                    
                    # Process only the first valid face found and break
                    break
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue
                
        return display_frame, probs, prediction_text