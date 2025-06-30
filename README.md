# Machine Learning Portoflio

This is a repository containing my Machine Learning Projects and is a record of my learning progress. I'm a Computer Science Student from Bina Nusantara University with a specialization in AI. This repository showcases my progress in learning machine learning and my past projects from simple ones, to the more advanced ones.

Linkedin Profile: https://www.linkedin.com/in/christian-luis-efendy-53b25a217/

# Linear Regression 


This project demonstrates a fundamental machine learning algorithm: simple linear regression. Implemented from scratch using Python's NumPy and Matplotlib libraries, it focuses on modeling the linear relationship between two variables.

The core of the project involves applying the least squares method to find the optimal slope and y-intercept for a line that best fits a given dataset. NumPy is used for efficient numerical computations, particularly for constructing the design matrix and solving the least squares equation using the pseudo-inverse (np.linalg.pinv). Matplotlib is then utilized to visualize the data points and the calculated regression line, providing a clear understanding of the model's fit.

This project serves as a foundational exercise in predictive modeling, illustrating the mathematical principles behind linear regression and the practical application of essential Python libraries for data analysis and visualization.

![LinearRegression](https://github.com/user-attachments/assets/86e6c444-4a8c-4572-92f9-23825ede5431)

# Logistic Regression 

This project delves into a practical application of machine learning: predicting diabetes using logistic regression. Leveraging the diabetes.csv dataset, I developed a classification model to determine the likelihood of a patient having diabetes based on various health indicators.

The project encompasses key stages of a typical machine learning workflow:

Data Loading and Initial Exploration: I used Pandas to load and get a first look at the dataset, understanding its structure and identifying columns with zero values that might require attention.

Feature Selection: Based on correlation analysis, I selected relevant features such as pregnant, glucose, insulin, bmi, pedigree, and age to train the model, excluding less impactful features like bp and skin.

Data Splitting: The dataset was split into training and testing sets to evaluate the model's performance on unseen data, ensuring robust evaluation.

Model Training: I utilized scikit-learn's LogisticRegression to train the classification model on the prepared training data.

Model Evaluation: The model's performance was assessed using standard classification metrics, including accuracy score, confusion matrix, and a detailed classification report providing precision, recall, and F1-score for both 'no_diabetes' and 'diabetes' classes.

# Clustering 

Customer Segmentation using K-Means, DBSCAN, and Hierarchical Clustering

This project explores customer segmentation using various unsupervised machine learning clustering algorithms: K-Means, DBSCAN, and Hierarchical Clustering. The goal is to identify distinct groups of customers within a mall dataset based on their demographics and spending habits, which can be valuable for targeted marketing strategies.

Project Highlights:

Data Exploration & Preprocessing: The Mall_Customers.csv dataset was loaded and inspected for initial insights. The 'Gender' column was transformed into a numerical format using Label Encoding for compatibility with clustering algorithms.

K-Means Clustering:

Elbow Method: The K-Elbow Visualizer from Yellowbrick was employed to determine the optimal number of clusters (k). Both distortion (sum of squared distances) and silhouette scores were used as metrics to find the "elbow" point, suggesting a suitable k.

Model Training & Visualization: A K-Means model was trained with the chosen number of clusters. The clustered data was then visualized using scatter plots of 'Annual Income (k$)' vs. 'Spending Score (1-100)' and 'Age' vs. 'Spending Score (1-100)', with clusters clearly distinguished by color and centroids marked.

Silhouette Score: The silhouette score for the K-Means model was calculated to evaluate the quality of the clusters, indicating how well-separated and cohesive the clusters are.

DBSCAN Clustering:

Parameter Optimization: A systematic approach was taken to find optimal eps (maximum distance between two samples for one to be considered as in the neighborhood of the other) and min_samples (number of samples in a neighborhood for a point to be considered as a core point) parameters. Heatmaps were generated to visualize the number of clusters and silhouette scores for various parameter combinations, aiding in the selection of the best-performing parameters.

Outlier Detection: DBSCAN's ability to identify outliers (noise points, labeled as -1) was utilized and visualized separately on the scatter plots.

Hierarchical Clustering:

Dendrogram Analysis: A dendrogram was generated to visually represent the hierarchical relationships between data points, helping to determine the optimal number of clusters by observing the longest vertical line that does not intersect any horizontal line.

Model Training & Visualization: An AgglomerativeClustering model was applied with the chosen number of clusters. Similar to K-Means, the clusters were visualized on scatter plots, showcasing the segmentation based on hierarchical grouping.

This project provides a comparative analysis of three prominent clustering techniques, demonstrating their application in customer segmentation and highlighting different approaches to cluster discovery and evaluation.
![Clustering](https://github.com/user-attachments/assets/ee5d0f39-2a86-4591-9a3d-0bbbad6621a9)

# Text Summarizer

This project focuses on developing text summarization capabilities specifically tailored for software engineering documents. Utilizing Python, I explored and implemented different approaches to automatically generate concise summaries from longer technical texts, which is invaluable for quickly grasping key information in documentation, research papers, or project specifications.

Project Highlights:

Preprocessing: Implemented techniques for cleaning and preparing text data, including tokenization, stop-word removal, and stemming/lemmatization, to ensure optimal input for the summarization models.

Extractive Summarization: Explored methods such as TF-IDF or TextRank to identify and extract the most important sentences directly from the original document.

This project demonstrates practical applications of natural language processing (NLP) in the software engineering domain, aiming to streamline information consumption and enhance productivity by automating the summarization of technical content.

# Sentiment Analysis

This project showcases a comprehensive data analysis and visualization workflow using a dataset related to "Satria Data," an Indonesian national data competition. The primary goal is to extract meaningful insights and present them through interactive and informative visualizations, providing a clear understanding of the competition's dynamics and participant demographics.

Project Highlights:

Data Loading and Initial Exploration: The project begins with loading the dataset using Pandas and performing initial exploratory data analysis (EDA) to understand its structure, identify data types, and check for missing values.

Data Cleaning and Preprocessing: Essential data cleaning steps are performed, such as handling missing values, correcting inconsistencies, and transforming data types as needed to ensure data quality for analysis.

Descriptive Statistics: Calculation of key descriptive statistics to summarize the main features of the dataset, including measures of central tendency and dispersion.

Categorical Data Analysis: Analysis of categorical variables, potentially including counts of participants by region, gender, or competition category.

Numerical Data Analysis: Exploration of numerical data, which might involve distributions of scores, ages, or other relevant quantitative metrics.

Data Visualization: Extensive use of Matplotlib, Seaborn, and potentially Plotly (if you used interactive plots) to create a variety of visualizations, such as:

Histograms and box plots for numerical distributions.

Bar charts for categorical data comparisons.

Scatter plots to explore relationships between variables.

This project demonstrates a foundational understanding of data analysis techniques, data preprocessing, and the crucial role of visualization in uncovering insights from raw data, especially within the context of a competitive event.
![sentiment_analysis](https://github.com/user-attachments/assets/efbe8ab5-7b4d-45d5-8ab6-34e9ab6df595)

# Penumonia Classification

This project focuses on building and evaluating a deep learning model for the classification of pneumonia from chest X-ray images. Leveraging the power of convolutional neural networks (CNNs), the aim is to develop an automated system that can assist in the early detection of pneumonia, a critical step in effective patient care.

Project Highlights:

Dataset: The project utilizes a dataset of chest X-ray images categorized as either "Pneumonia" or "Normal."

Image Preprocessing: Implemented various image preprocessing techniques, including resizing, normalization, and augmentation (e.g., rotation, zoom, shear), to prepare the images for the neural network and enhance model generalization.

Convolutional Neural Network (CNN) Architecture: Designed and implemented a CNN model tailored for image classification tasks. The architecture includes convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.

Model Training: The CNN model was trained on the preprocessed X-ray images, with careful monitoring of training and validation loss/accuracy to prevent overfitting.

Model Evaluation: Assessed the model's performance using key metrics such as accuracy, precision, recall, and F1-score. A confusion matrix was generated to provide a detailed breakdown of correct and incorrect classifications.

Visualization of Results: Visualized training progress and model predictions to gain insights into the model's learning behavior and performance on unseen data.

This project demonstrates a practical application of deep learning in medical image analysis, highlighting the potential of AI to support healthcare diagnostics.

![Chest_Xray](https://github.com/user-attachments/assets/ac1a7f5f-d209-4bf9-8929-e3fa9e230e28)

# AI and Plagiarism Text Detector

This project focuses on developing a machine learning model to detect whether a given text was generated by an AI or written by a human. As AI-generated content becomes more prevalent, the ability to distinguish between human and synthetic text is increasingly important for various applications, including content moderation, academic integrity, and disinformation detection.

Project Highlights:

Dataset: The project utilizes a dataset containing examples of both human-written and AI-generated texts.

Text Preprocessing: A crucial step in this project involved comprehensive text preprocessing. This included tokenization, stop-word removal, and normalization techniques to prepare the text data for effective feature extraction and model training.

Feature Engineering: Various NLP techniques were applied to extract meaningful features from the processed text. This likely involved approaches such as Bag-of-Words, TF-IDF (Term Frequency-Inverse Document Frequency), or potentially more advanced methods like word embeddings to capture stylistic and linguistic patterns unique to AI-generated content.

Model Selection and Training: Different machine learning classifiers were explored and trained on the engineered features. Common choices for this task include Naive Bayes, Support Vector Machines (SVMs), Logistic Regression, or ensemble methods like Random Forests and Gradient Boosting.

Model Evaluation: The performance of the AI text detection model was rigorously evaluated using standard classification metrics. This typically included accuracy, precision, recall, and F1-score. A confusion matrix was also likely generated to provide a detailed breakdown of true positives, true negatives, false positives, and false negatives.

This project demonstrates practical skills in natural language processing and machine learning for a relevant and emerging real-world problem, showcasing an understanding of text classification and model evaluation for AI-generated content.

# DeepFake Audio Detection

This project focuses on developing a machine learning model to detect manipulated or synthetic audio, commonly known as "audio DeepFakes." As audio synthesis technology advances, the ability to discern authentic human speech from AI-generated or altered audio is becoming increasingly vital for security, verification, and combating misinformation.

Project Highlights:

Dataset: The project utilizes a dataset comprising both authentic (real) and DeepFake audio samples, crucial for training a robust detection model.

Audio Preprocessing: A fundamental step involved preparing the raw audio data. This included techniques such as downsampling, normalization, and converting audio signals into suitable representations for analysis, such as Mel-Frequency Cepstral Coefficients (MFCCs) or spectrograms.

Feature Extraction: Advanced audio feature extraction methods were employed to capture subtle characteristics that differentiate real from synthetic speech. This goes beyond basic waveform analysis to identify anomalies introduced by manipulation techniques.

Deep Learning Model Architecture: A deep learning model, likely a Convolutional Neural Network (CNN) or a Recurrent Neural Network (RNN, such as LSTM or GRU), or a hybrid architecture, was designed and implemented. These models are effective at learning complex patterns from sequential and spectral audio features.

Model Training and Optimization: The model was trained on the preprocessed audio features. Emphasis was placed on optimizing the training process, including hyperparameter tuning and regularization techniques, to achieve high detection accuracy and generalize well to unseen audio.

Performance Evaluation: The model's efficacy in identifying audio DeepFakes was thoroughly evaluated using standard classification metrics. This typically included accuracy, precision, recall, and F1-score, providing a comprehensive understanding of its detection capabilities.

This project demonstrates practical skills in digital signal processing, deep learning for audio analysis, and addresses a significant challenge in ensuring the authenticity and integrity of audio content.

![DeepFakeDetection](https://github.com/user-attachments/assets/243d805d-76ec-4b78-8a46-152c9752dd54)

# Image Classification: A Performance Comparison of Vision Transformers, EfficientNets, and Hybrid Architectures

This research-oriented project investigates and compares the performance of different deep learning architectures for image classification tasks, with a particular focus on understanding the benefits and trade-offs of standalone Vision Transformers (ViT), EfficientNets, and hybrid models. The core objective is to analyze how these diverse architectural approaches handle visual data and their impact on classification accuracy and efficiency.

Project Highlights:

Architectural Exploration:

Vision Transformers (ViT): Implemented and experimented with Vision Transformers, a novel architecture that applies the Transformer model (originally for NLP) directly to image patches.

EfficientNets: Explored EfficientNet models, which utilize compound scaling to uniformly scale network depth, width, and resolution, leading to highly efficient and accurate CNNs.

Hybrid Architectures: Developed and tested hybrid models that combine elements of CNNs (like EfficientNet) with Transformers (like ViT), aiming to leverage the strengths of both for superior performance.

Performance Comparison: A key aspect of the project is the systematic comparison of these architectures. This involved training each model on a common image dataset and evaluating their performance based on metrics such as accuracy, loss, and potentially inference speed.

Experimental Design for Research: The project is structured with a clear experimental design to facilitate a rigorous comparison. This includes consistent data preprocessing, training protocols, and evaluation methodologies across all architectural variants.

Insight into Architectural Strengths: The research aims to provide insights into which architectural components are most effective for specific image classification challenges and whether hybrid approaches offer significant advantages over standalone models.

Visualization of Results: Visualizations of training curves, confusion matrices, and comparative performance plots are used to illustrate the findings and support the conclusions drawn from the experiments.

This project demonstrates strong research skills in deep learning, focusing on advanced image classification architectures and providing a comparative analysis of their efficacy and design principles.

![vit-efficientnetb1](https://github.com/user-attachments/assets/7d88d516-519f-4eca-96ee-9b35a167aa2b)

# Face Shape Classification

This project develops an AI-powered system for classifying face shapes from images, with a practical application in mind for a mobile app. The core of this system involves leveraging computer vision techniques to extract facial landmarks and then using these landmarks to train a machine learning model capable of accurately identifying different face shapes. The project also demonstrates the fundamental steps required for integrating such AI capabilities into an iOS application.

Project Highlights:

Facial Landmark Extraction (iOS & Python):

Python (Jupyter Notebook): The Python notebook details the process of using computer vision libraries to detect facial landmarks from input images. This typically involves identifying key points like the corners of the eyes, nose, mouth, and jawline.

Swift (LandmarkGenerator.swift): The Swift file showcases how facial landmark detection can be performed directly on an iOS device. This code likely utilizes Apple's Vision framework to detect faces and their associated landmarks, providing the necessary data for real-time app integration.

Face Shape Classification Model:

Feature Engineering: Features are engineered from the extracted facial landmarks. This might include distances between specific points, angles, ratios, or other geometric properties that are indicative of different face shapes (e.g., oval, round, square, heart, long).

Machine Learning Classification: A machine learning model (e.g., SVM, Random Forest, or a simple Neural Network) is trained on these engineered features to classify images into predefined face shape categories. The Jupyter Notebook outlines the model training, evaluation, and hyperparameter tuning processes.

Performance Evaluation: The classification model's performance is evaluated using standard metrics such as accuracy, precision, recall, and F1-score to ensure its reliability in correctly identifying face shapes.

Application-Oriented Design: The dual-language approach (Python for model development and Swift for iOS integration) highlights a practical pipeline for deploying AI models into real-world applications. The project focuses on building a functional AI component that can serve as a backend for a mobile app feature.

This project demonstrates expertise in computer vision, machine learning for classification, feature engineering from visual data, and foundational steps for iOS app integration, all applied to the engaging problem of face shape analysis.

![face_shape_detector](https://github.com/user-attachments/assets/1d062da0-f6a9-4689-90cf-b1d98f69c1c7)

# Pose to Impress

Link External Github Repo: https://github.com/LuisSsiuL/pose-to-impress

This project develops an AI-driven system that provides real-time feedback and guidance on human pose, aiming to help users achieve specific postures or improve their form. It integrates computer vision for pose estimation with a backend system for pose analysis and potentially user feedback. This project has applications in fitness, dance, rehabilitation, or any scenario requiring precise body positioning.

Project Highlights:

Real-time Pose Estimation: The system utilizes computer vision techniques to detect and track key human body landmarks in real-time. Files like phone_camera.py and easy_main.py suggest an emphasis on live video input and efficient processing.

Pose Analysis and Scoring: Beyond simple detection, the project likely includes logic to analyze the detected poses against predefined target poses. This could involve comparing joint angles, limb orientations, or spatial relationships between landmarks. The pose_coordinates.json file would define these target poses.

Modular Architecture: The project is structured with a modular design, indicated by files such as create_outline.py, create_output.py, and main.py. This suggests a clear separation of concerns for tasks like pose processing, feedback generation, and overall application flow.

User Guidance and Feedback: The create_output.py script implies a mechanism for generating actionable feedback or instructions for the user, guiding them towards the correct pose. This might be visual (overlaying correct posture, as suggested by frame.png) or textual.

Python for AI/CV Core: Python scripts (e.g., main.py, easy_main.py, phone_camera.py, create_outline.py, create_output.py) form the backbone of the AI and computer vision components, likely utilizing libraries such as OpenCV and MediaPipe for pose detection.

Application-Oriented Development: The naming conventions and inclusion of phone_camera.py indicate a strong focus on building a system that can be integrated into mobile or webcam-based applications, providing interactive pose correction.

This project demonstrates strong skills in real-time computer vision, human pose estimation, and the application of AI for interactive user guidance.

![20241114163346](https://github.com/user-attachments/assets/fbbc6e19-845d-4925-9bdc-748c25b20d62)



