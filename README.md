# Type 2 Diabetes Risk Prediction App

## Overview
This project provides a robust pipeline and interactive web app for predicting Type 2 Diabetes Mellitus (T2DM) risk using clinical and lifestyle data. It features:
- Data ingestion and cleaning
- Feature engineering and selection
- Model training with multiple algorithms
- Automated model selection
- SHAP-based interpretability
- A Streamlit app for user-friendly risk prediction
- Multi-model support and prediction logging in a database

## Project Structure (Best Practice)
```
diabetes-risk/
│
├── data/                    # (Optional) For raw, interim, and processed data
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/                  # All model artifacts (pkl files from Colab, etc.)
│   ├── model1_model.pkl
│   ├── model1_preprocessor.pkl
│   ├── model1_feature_selector.pkl
│   ├── model1_feature_columns.pkl
│   └── ...
│
├── db/                      # Database files
│   └── predictions.db
│
├── src/                     # All Python source code (logic, utils, etc.)
│   ├── __init__.py
│   ├── train_and_save.py    # Model training script
│   ├── db_utils.py          # Database utility functions
│   └── ...
│
├── app/                     # Streamlit and web app code
│   └── t2dm_app.py          # Main Streamlit app
│
├── notebooks/               # Jupyter notebooks for EDA, SHAP, etc.
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Setup
1. Clone this repository or copy the files to your working directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your data files (e.g., `integer_no_arrays.tsv`, `real_fields1.tsv`) in the `data/` folder (optional).
4. Place all model artifacts (`*_model.pkl`, etc.) in the `models/` directory.
5. The database (`predictions.db`) will be created in the `db/` directory.

## Training the Model
Run the training script to process data, train models, select the best one, and save all necessary artifacts:
```bash
python src/train_and_save.py
```
Artifacts saved:
- `models/modelname_model.pkl` (the best classifier)
- `models/modelname_preprocessor.pkl` (fitted scaler/encoder)
- `models/modelname_feature_selector.pkl` (fitted feature selector)
- `models/modelname_feature_columns.pkl` (list of numeric and categorical columns)

## Running the Streamlit App
Start the app with:
```bash
streamlit run app/t2dm_app.py
```
- Select a model for prediction from the dropdown
- Enter your data manually for instant risk prediction
- View SHAP explanations for transparency
- Recent predictions are displayed and logged in the database

## Extending the App
- Add new features by updating the training script and retraining
- Adjust model selection or add new algorithms as needed
- Customize the Streamlit UI for additional user guidance
- Add notebooks for EDA or advanced SHAP analysis in `notebooks/`

## SHAP Interpretability
The app provides SHAP-based explanations for each prediction, helping users understand the most influential features.

## License
MIT License 