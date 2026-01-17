import pandas as pd
import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def train_model():
    print("🚀 Fetching UCI Iranian Churn dataset...")
    # Fetch dataset
    iranian_churn = fetch_ucirepo(id=563)
    
    # Data (as pandas dataframes)
    X = iranian_churn.data.features
    y = iranian_churn.data.targets
    
    # Flatten y if it's a dataframe
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    print(f"✅ Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")

    # 80/20 train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model Hyperparameters
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )

    # 10-fold cross-validation on training set
    print("🔄 Running 10-fold cross-validation...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"📊 CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    # Train on full training set
    model.fit(X_train, y_train)

    # Test set performance
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"📈 Test Set Accuracy: {test_accuracy:.4f}")
    print("\n📝 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n🧱 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Verify target accuracy (96.39% ± 1.08%)
    target_min = 0.9639 - 0.0108
    if cv_mean >= target_min:
        print(f"✅ Target accuracy achieved! ({cv_mean:.4f} >= {target_min:.4f})")
    else:
        print(f"⚠️ Target accuracy not achieved. ({cv_mean:.4f} < {target_min:.4f})")

    # Save artifacts to backend/
    print("💾 Saving artifacts to backend/...")
    os.makedirs('backend', exist_ok=True)
    joblib.dump(model, 'backend/churn_model.pkl')
    joblib.dump(X.columns.tolist(), 'backend/feature_names.pkl')
    
    metadata = {
        'cv_accuracy': cv_mean,
        'cv_std': cv_std,
        'test_accuracy': test_accuracy,
        'feature_count': len(X.columns),
        'model_type': 'LightGBM'
    }
    joblib.dump(metadata, 'backend/model_metadata.pkl')
    print("✅ All artifacts saved successfully.")

if __name__ == "__main__":
    train_model()
