# Bio-Signal Analysis for Smoking Classification

This project presents a complete end-to-end machine learning pipeline for predicting smoking behavior based on bio-signal and physiological health data. The solution leverages statistical analysis, feature engineering, and ensemble learning to classify individuals as smokers or non-smokers using real-world health metrics.

## Problem Statement

The objective of this project was to develop an intelligent classification system that can infer the smoking status of an individual based solely on their biological signals. This can assist healthcare professionals in non-invasive diagnosis and risk assessment without requiring a direct declaration of smoking habits.

## Dataset Description

The dataset comprises over 55,000 samples and 27 health-related features, including demographic details (age, gender), anthropometric measures (height, weight, waist), vital signs (blood pressure), vision, hearing, and multiple laboratory test results such as cholesterol levels, blood sugar, liver enzymes, and hemoglobin.

### Key Features:
- `age`, `gender`, `height(cm)`, `weight(kg)`, `waist(cm)`
- `systolic`, `relaxation` (blood pressure), `fasting blood sugar`, `Cholesterol`, `HDL`, `LDL`, `triglyceride`
- `hemoglobin`, `AST`, `ALT`, `Gtp`, `serum creatinine`
- `oral`, `tartar`, `dental caries`, `eyesight(left/right)`, `hearing(left/right)`
- Target variable: `smoking` (binary classification - smoker or non-smoker)

## Project Workflow

### 1. Data Preprocessing
- Removal of irrelevant features (`ID`)
- Handling of categorical variables (`gender`, `oral`, `tartar`, etc.)
- Conversion of binary string features (`Y/N`) to numeric
- One-hot encoding for categorical attributes
- Missing value analysis and treatment

### 2. Exploratory Data Analysis (EDA)
- Univariate and bivariate distributions
- Visual insights on gender-wise and age-wise smoking patterns
- Boxplots for identifying outliers (true outliers retained due to natural variation)

### 3. Feature Selection
- Employed `ExtraTreesClassifier` to compute feature importances
- Selected top 15 most relevant features based on importance scores to reduce dimensionality and enhance model performance

### 4. Model Development
Implemented and evaluated the following models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Bagging Classifier**
- **Random Forest Classifier**
- **Extra Trees Classifier**

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, and F1-Score computed for each model
- Bagging Classifier demonstrated the best performance with approximately 82.6% accuracy and strong precision/recall balance for the smoker class

### 6. Model Serialization
- Final Bagging Classifier and StandardScaler objects were serialized using `joblib` for future inference and deployment

### 7. Interactive Deployment
- Deployed a fully functional **Gradio-based web interface** that allows users to input health parameters and receive real-time classification output (Smoker or Non-Smoker)

## Technical Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib, gradio, joblib
- **Modeling:** Ensemble Learning (Bagging, Random Forest, Extra Trees)
- **Deployment:** Gradio UI for interactive web-based inference

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/surendiran-20cl/bio-signal-smoking-classification.git
   cd bio-signal-smoking-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (if not already saved):
   ```bash
   python train_model.py
   ```

4. Launch the Gradio app:
   ```bash
   python gradio_app.py
   ```

## Use Cases and Applications

- Health risk stratification tools
- Early warning systems in preventive healthcare
- Assistive diagnosis in digital health solutions
- Population-level health behavior studies

## Project Outcomes

- Achieved over 82% accuracy with ensemble methods
- Created a compact feature space with only 15 top-ranked attributes
- Built a reusable and interactive Gradio interface for model inference
- Prepared for extension into deployment with API endpoints or cloud hosting

## Author

This project was developed as part of an Artificial Intelligence course capstone project to demonstrate applied knowledge in health informatics using machine learning.

## License

This project is released under the Apache 2.0 License.
