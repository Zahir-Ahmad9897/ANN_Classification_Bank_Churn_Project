# Bank Customer Churn Prediction (ANN)

This project utilizes an Artificial Neural Network (ANN) to classify bank customers based on their likelihood to churn (leave the bank). It features a Streamlit-based web application for interactive predictions.

## 📂 Project Structure

- **`app.py`**: The main Streamlit application for real-time predictions.
- **`ANN_Project1_BankDataset.ipynb`**, **`ANN_Prediction.ipynb`**: Jupyter notebooks used for training and testing the model.
- **`Churn_Modelling.csv`**: The dataset used for training.
- **`Regression_model.h5`**: The trained Keras/TensorFlow model.
- **`preprocessing artifacts`**: `label_encoder_gender.pkl`, `one_hot_encoder.pkl`, `scaler.pkl` used for data transformation.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A virtual environment is recommended.

### Installation

1.  **Clone the repository** (if applicable):
    ```bash
    git clone <repository-url>
    cd ANN_Classification_Bank_Churn_Project
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## 🧠 Model Details

- **Type**: Artificial Neural Network (Classification)
- **Input Features**: Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary.
- **Output**: Probability of churn (0 to 1). Using a threshold of 0.5 for classification.

## ⚠️ Note

Ensure that `Regression_model.h5`, `label_encoder_gender.pkl`, `one_hot_encoder.pkl`, and `scaler.pkl` are present in the same directory as `app.py` for the application to run correctly.