⚽ **Player Face Identification (Streamlit + OpenCV)**

Detect a face in a photo and identify which player it is using:

MTCNN (face detection + alignment)

FaceNet (InceptionResnetV1) embeddings

A simple classifier (SVM or k-NN) trained on your images

Open-set handling: mark faces as Unknown using probability & cosine-similarity thresholds

✨ **Features**

Upload an image or use your camera

Detect multiple faces and label each

“Unknown” fallback for untrained faces (thresholds configurable)

CPU-friendly; GPU not required to run the app

Small, simple codebase



**Tech Stack**

Python, Streamlit

OpenCV (I/O + drawing)

facenet-pytorch (MTCNN + FaceNet embeddings)

scikit-learn (SVM or k-NN)

PyTorch (model runtime)


**Project Structure**
player-face-id/
|-- app.py
|-- requirements.txt
`-- models/
    |-- player_id.joblib
    |-- labels.joblib
    `-- embeddings.npz

