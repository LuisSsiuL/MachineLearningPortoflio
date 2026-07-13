# Machine Learning Portfolio

Applied machine learning and computer vision projects spanning medical imaging, biometric verification, NLP, audio processing, and mobile integration.

## Overview

I am a Computer Science student at Bina Nusantara University specializing in Artificial Intelligence. This repository collects my machine learning projects — each directory contains a self-contained notebook with code and result plots, and two related projects are linked out to their own repositories. The emphasis throughout is on building deployable, real-world AI solutions rather than toy examples.

LinkedIn: https://www.linkedin.com/in/christian-luis-efendy-53b25a217/

## Featured Projects

### Skin Disease Detection with a Hybrid CNN-Transformer (Undergraduate Thesis)

<img width="1189" height="490" alt="vit-efficientnetb1" src="https://github.com/user-attachments/assets/804e08c1-a36e-43b8-a019-f690e75002fc" />

Classifies skin lesions in dermatoscopic images using a hybrid architecture that combines an EfficientNet-B1 CNN for local feature extraction with a Vision Transformer for global context. Designed for real-world diagnostic assistance and published in a scientific journal.
Directory: `ViT-EfficientNet Hybrid/`

### IPC Similarity Verifier - Face Matching and Liveness Detection

Verifies that the person in a selfie matches the photo on an uploaded KTP (Indonesian ID card) and runs a liveness (anti-spoofing) check. Built with Flask, OpenCV, DeepFace, and Gunicorn. All images are processed in memory only — nothing is stored or collected.

Live demo (Hugging Face): https://c-luis-e-ipc-similarity-verifier.hf.space

- `POST /api/verify` — multipart `ktp_image` + `selfie_image`; returns `matched`, `distance`, `threshold`, and a message
- `POST /api/liveness` — multipart `image`; returns `liveness_passed` and a confidence score

### Wise Frame - Face and Eyewear Matching (iOS)

<img width="1489" height="1189" alt="face_shape_detector" src="https://github.com/user-attachments/assets/c7cc456d-5ea2-4176-944b-eb7486891ce0" />

Mobile app that recommends eyeglass frames from face shape, skin tone, and facial proportions. A Python ML model (facial landmark extraction for face-shape detection) is integrated into a native SwiftUI app with an ARKit virtual try-on, product listings, and onboarding. Stack: SwiftUI, ARKit, Vision, MediaPipe, Python, CoreML.
The `Facial Shape Detector/` directory holds the face-shape model notebook and the Swift landmark generator.
App repository: https://github.com/celine1906/C8S2-MLChallenge-WiseFrame

### Pose to Impress

Real-time pose correction from webcam or mobile input for fitness and dance posture tracking, built with OpenCV and MediaPipe.
Repository: https://github.com/LuisSsiuL/pose-to-impress

### Pneumonia Detection from Chest X-Rays

<img width="1567" height="1582" alt="Chest_Xray" src="https://github.com/user-attachments/assets/49bb545c-1173-4966-88ff-ed8a5a806cc6" />

CNN classifier labelling chest X-rays as Normal or Pneumonia, trained with image augmentation and evaluated with standard classification metrics.
Directory: `Pneumonia Classification/`

### DeepFake Audio Detection

<img width="1033" height="470" alt="DeepFakeDetection" src="https://github.com/user-attachments/assets/ecb74e9d-c6d3-48af-8316-90f6515ff8d5" />

Deep learning system that detects synthetic audio using CNNs over audio features such as MFCCs and spectrograms.
Directory: `Deep Fake Detection/`

### AI Text Detector and Plagiarism Checker

Two-part NLP pipeline: a stacked-LSTM network trained on labelled sentence pairs for plagiarism detection, and an AI-vs-human text classifier that constructs its own dataset (GPT-2 generations seeded with Project Gutenberg prompts) and classifies SBERT sentence embeddings with Multinomial Naive Bayes.
Directory: `AI Text Detector/`

### Scientific Paper Summarizer

Extractive summarizer tailored to software engineering documents, built on TF-IDF and TextRank with custom NLP preprocessing.
Directory: `Scientific Paper Summarizer/`

### Satria Data Competition Insights

<img width="567" height="455" alt="sentiment_analysis" src="https://github.com/user-attachments/assets/ed7fc982-c0fb-4c9a-80fb-b41866276464" />

Exploratory analysis of participant demographics and trends in Indonesia's Satria Data competition, using Matplotlib, Seaborn, and Plotly.
Directory: `Sentiment Analysis/`

### Customer Segmentation with Clustering

Segments mall customers by demographics and spending habits using K-Means (Elbow Method, Silhouette Score), DBSCAN for outlier detection, and hierarchical clustering with dendrograms.
Directory: `AI Clustering/`

### Classical ML Foundations

- Diabetes prediction — logistic regression binary classifier over health indicators (`Logistic Regression/`)
- Linear regression implemented from scratch with NumPy and visualized with Matplotlib (`Linear Regression/`)

## Tech Stack

Python, TensorFlow/Keras, PyTorch, scikit-learn, OpenCV, MediaPipe, Hugging Face Transformers, Sentence-Transformers, NLTK, Flask, Matplotlib, Seaborn, Plotly, and (for Wise Frame) SwiftUI, ARKit, and CoreML.

## Repository Structure

Each project lives in its own directory with its notebook and a result snapshot:

- `ViT-EfficientNet Hybrid/` — thesis: hybrid CNN-Transformer skin lesion classifier
- `Facial Shape Detector/` — Wise Frame face-shape model and Swift landmark generator
- `Pneumonia Classification/`, `Deep Fake Detection/` — medical imaging and audio deepfake CNNs
- `AI Text Detector/`, `Scientific Paper Summarizer/`, `Sentiment Analysis/` — NLP and analytics
- `AI Clustering/`, `Logistic Regression/`, `Linear Regression/` — classical ML
