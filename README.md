# Intrusion Detection System Tester for DDoS Attack Detection

This project is a machine-learning-based Intrusion Detection System (IDS) tester designed to detect Distributed Denial of Service (DDoS) attacks. It evaluates and compares the performance of three different machine learning models—Neural Network, Random Forest, and Graph Neural Network (GNN)—to identify the best approach for DDoS detection. Additionally, it includes a web application for visualizing data and results.

## Files Overview

### **1. Data preprocesing**
This script handles data upload and preprocessing tasks. It ensures the dataset is cleaned, normalized, and prepared for training and testing. The dataset we are using on this project is Aposemat-IoT23

File: variables_selection.ipynb

- **Functions**:
  - Load raw dataset.
  - Perform feature scaling and encoding.
  - Save the processed dataset for use in training models.


### **2. Neural Network**
This script trains and tests a Neural Network model.

- **Features**:
  - Configurable hyperparameters (e.g., number of layers, activation functions).
  - Outputs performance metrics such as accuracy, precision, recall, and F1-score.
  - Saves the trained model for future use.

File: nn_model.ipynb


### **3. Random Forest**
This script trains and tests a Random Forest model.

- **Features**:
  - Allows customization of the number of trees and depth.
  - Evaluates model performance using standard metrics.
  - Saves the trained model for further evaluation.

File: random_forest.ipynb

### **4. Graph Neural Network**
This script trains and tests a Graph Neural Network (GNN).

- **Features**:
  - Processes graph-structured datasets.
  - Implements GNN layers to identify patterns in the data.
  - Outputs comparative metrics for evaluation.

File: gnn.py

### **5. Web Application**
A web application that provides an interactive interface for exploring data and visualizing results.

- **Features**:
  - Upload and view processed datasets.
  - Visualize model performance (e.g., charts, graphs).
  - Compare results across the three models.

File: app.py

Start the web app using 
```
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`
