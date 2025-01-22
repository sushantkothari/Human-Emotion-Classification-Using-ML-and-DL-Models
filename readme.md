# Emotion Classification using ML and DL

## Overview
This project implements both machine learning and deep learning models to classify text data into different emotion categories. Leveraging LSTMs for deep learning and various machine learning algorithms, the project demonstrates a comprehensive approach to emotion classification.

---

## Dataset
The dataset used contains text samples labeled with emotions such as happiness, sadness, anger, fear, etc. The data is preprocessed and split into training and testing sets to evaluate the performance of the models.

---

## Project Structure
The project is divided into the following steps:

1. Dependency Installation
   - Installing required libraries and frameworks such as TensorFlow, Keras, Scikit-learn, Pandas, and NLTK.

2. Data Loading
   - Importing and loading the dataset into the environment.

3. Exploratory Data Analysis (EDA)
   - Analyzing the distribution of emotions and the characteristics of the text data.

4. Data Preprocessing
   - Encoding emotions into numerical values.
   - Cleaning text data (removal of punctuation, stopwords, etc.).
   - Tokenizing and padding sequences for deep learning models.

5. Machine Learning Models
   - Training and evaluating various machine learning algorithms for emotion classification.

6. Deep Learning Model (LSTM)
   - Implementing an LSTM-based model architecture.
   - Training and evaluating the LSTM model.

7. Predictive System
   - Building a system to predict emotions from new text inputs.

8. Model Saving
   - Saving the trained models and preprocessing files for future use.

---

## Key Features
- Utilizes LSTM for deep learning and machine learning algorithms for comparative analysis.
- Implements robust text preprocessing techniques, including:
  - Tokenization and word vectorization.
  - Padding sequences to ensure uniform input size for the models.
- Provides detailed evaluation metrics for each model.
- Visualizes performance through confusion matrices and classification reports.
- Saves trained models, allowing for deployment and reuse in real-world applications.

### Construction
The construction of the project focuses on creating a modular and extensible workflow:

1. Data Handling: Includes loading, cleaning, and preparing text data for model training.
2. Model Implementation:
   - Machine Learning Models: Includes Multinomial Naive Bayes, Logistic Regression, Random Forest, Support Vector Machine, and AdaBoost.
   - Deep Learning Model: An LSTM-based network architecture is implemented for sequential text data.
3. Evaluation Framework: Consists of metrics such as accuracy, precision, recall, and F1-score.
4. Prediction Pipeline: Allows users to input new text data and get emotion predictions using the best-performing model.

---

## Classification Metrics for All Models
| Model                     | Accuracy | Precision | Recall  | F1-score |
|---------------------------|----------|-----------|---------|----------|
| Multinomial Naive Bayes   | 0.655000 | 0.757804  | 0.655000| 0.581576 |
| Logistic Regression       | 0.853437 | 0.851707  | 0.853437| 0.850055 |
| Random Forest             | 0.850625 | 0.850188  | 0.850625| 0.848949 |
| Support Vector Machine    | 0.858750 | 0.857442  | 0.858750| 0.856062 |
| AdaBoost                  | 0.328437 | 0.493747  | 0.328437| 0.176962 |

### Best Model Based on Accuracy
| Metric       | Support Vector Machine |
|--------------|-------------------------|
| Accuracy     | 0.85875                 |
| Precision    | 0.857442                |
| Recall       | 0.85875                 |
| F1-score     | 0.856062                |

---

## Installation
To set up the environment, install the necessary dependencies:

bash
pip install tensorflow==2.15.0 scikit-learn pandas numpy seaborn matplotlib wordcloud nltk


---

## Usage
To run the project:

1. Clone the repository:
   bash
   git clone https://github.com/sushantkothari/Human-Emotion-Classification-Using-ML-and-DL-Models.git
   

2. Open the Jupyter notebook:
   bash
   jupyter notebook FInal_Final_Emotions_Classification_using_ML_and_DL.ipynb
   

4. Follow the instructions in the notebook to run each section sequentially.

---

## Model Architectures
### Machine Learning Models
- Algorithms like Logistic Regression, Random Forest, and Support Vector Machines were trained and evaluated.

### Deep Learning Model (LSTM)
The deep learning model architecture includes:
- Embedding Layer: Converts words into dense vectors.
- LSTM Layer: Captures sequential dependencies in the text.
- Dropout: Prevents overfitting.
- Dense Layers: Maps features to emotion categories.

---

## Final Results

The deep learning model significantly outperformed the machine learning models, achieving an accuracy of 95.68%, compared to the highest accuracy of 85.87% obtained by the Support Vector Machine model. This highlights the superior capability of LSTM in capturing sequential dependencies and contextual information from text data.

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgements
Special thanks to TensorFlow, Keras, and Scikit-learn communities for their tools and frameworks that made this project possible.
