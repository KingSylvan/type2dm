import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings("ignore")

# --- Load & Clean Data ---
def load_and_process_data():
    df1 = pd.read_csv("integer_no_arrays.tsv", sep="\t")
    df2 = pd.read_csv("real_fields1.tsv", sep="\t")
    df = pd.concat([df1, df2], axis=1)
    print(f"✅ Combined dataset shape: {df.shape}")
    return df

def clean_and_preprocess(df):
    df = df[df['age'] >= 18]
    df.dropna(axis=1, thresh=len(df)*0.9, inplace=True)
    df.dropna(inplace=True)
    return df

def create_target_variable(df):
    df['t2dm'] = ((df['hba1c'] > 6.5) | (df['glucose'] > 125)).astype(int)
    print(f"✅ T2DM prevalence: {df['t2dm'].mean():.2%}")
    return df

def prepare_features(df):
    X = df.drop(columns=['t2dm'])
    y = df['t2dm']
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return X, y, numeric_features, categorical_features

def create_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_pipeline = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    return preprocessor

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

def select_top_features(X_train, y_train, X_test, k=30):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector

def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_pr = average_precision_score(y_test, y_prob)
        results[name] = {'model': model, 'auc_pr': auc_pr}
    return results

def main():
    print("\n🏥 Training Type 2 Diabetes Prediction Model...")
    df = load_and_process_data()
    df = clean_and_preprocess(df)
    df = create_target_variable(df)
    X, y, numeric_features, categorical_features = prepare_features(df)
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    X_train_resampled, y_train_resampled = apply_smote(X_train_proc, y_train)
    X_train_selected, X_test_selected, selector = select_top_features(X_train_resampled, y_train_resampled, X_test_proc)
    results = train_models(X_train_selected, y_train_resampled, X_test_selected, y_test)
    best_model_name = max(results, key=lambda x: results[x]['auc_pr'])
    best_model = results[best_model_name]['model']
    print(f"\n🏆 Best model: {best_model_name} (AUC-PR: {results[best_model_name]['auc_pr']:.3f})")
    # Save artifacts
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(selector, "feature_selector.pkl")
    joblib.dump((numeric_features, categorical_features), "feature_columns.pkl")
    print("\n✅ Artifacts saved: best_model.pkl, preprocessor.pkl, feature_selector.pkl, feature_columns.pkl")

if __name__ == "__main__":
    main() 