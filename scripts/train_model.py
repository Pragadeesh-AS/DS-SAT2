import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate_models():
    print("=" * 65)
    print("STAYNEST HOSPITALITY - PREDICTIVE CANCELLATION MODEL TRAINING")
    print("Course Outcomes: CO3 (Predictive Modeling) & CO4 (Feature Engineering)")
    print("=" * 65)

    data_path = "cleaned_data/staynest_cleaned_data.csv"
    print(f"\n1. Ingesting cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Feature selection for modeling
    features = [
        'lead_time', 'total_stay_nights', 'rate', 'total_guests',
        'total_of_special_requests', 'required_car_parking_spaces',
        'room_type_mismatch', 'is_repeated_guest', 'previous_cancellations',
        'booking_changes', 'deposit_type', 'booking_channel',
        'guest_type', 'market_segment', 'season', 'has_event'
    ]

    X = df[features].copy()
    y = df['is_canceled'].copy()

    # Categorical encoding
    cat_cols = ['deposit_type', 'booking_channel', 'guest_type', 'market_segment', 'season']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # 80/20 Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n2. Train-Test Split: Train Size = {len(X_train):,}, Test Size = {len(X_test):,}")

    # Model 1: Baseline Logistic Regression
    print("\n3. Training Baseline Model: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, lr_probs)
    print(f"   -> Logistic Regression Accuracy: {lr_acc*100:.2f}% | ROC-AUC: {lr_auc:.4f}")

    # Model 2: Random Forest Classifier
    print("\n4. Training Advanced Model: Random Forest Classifier (150 trees)...")
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)

    print("\n" + "=" * 65)
    print("MODEL EVALUATION COMPARISON RESULTS")
    print("=" * 65)
    comparison_df = pd.DataFrame([
        {'Model': 'Logistic Regression (Baseline)', 'Accuracy': f"{lr_acc*100:.2f}%", 'Precision': f"{precision_score(y_test, lr_preds)*100:.2f}%", 'Recall': f"{recall_score(y_test, lr_preds)*100:.2f}%", 'F1-Score': f"{f1_score(y_test, lr_preds):.4f}", 'ROC-AUC': f"{lr_auc:.4f}"},
        {'Model': 'Random Forest Classifier (Best)', 'Accuracy': f"{rf_acc*100:.2f}%", 'Precision': f"{rf_prec*100:.2f}%", 'Recall': f"{rf_rec*100:.2f}%", 'F1-Score': f"{rf_f1:.4f}", 'ROC-AUC': f"{rf_auc:.4f}"}
    ])
    print(comparison_df.to_string(index=False))

    print("\nDetailed Classification Report for Random Forest:")
    print(classification_report(y_test, rf_preds, target_names=['Completed Stay (0)', 'Canceled (1)']))

    cm = confusion_matrix(y_test, rf_preds)
    print("Confusion Matrix:")
    print(f"True Negatives:  {cm[0, 0]:,} (Correctly identified completed stays)")
    print(f"False Positives: {cm[0, 1]:,} (Predicted cancel, but stayed)")
    print(f"False Negatives: {cm[1, 0]:,} (Predicted stay, but canceled)")
    print(f"True Positives:  {cm[1, 1]:,} (Correctly caught cancellations)")

    # Save trained model artifact
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_artifact = {
        'model': rf_model,
        'features': features,
        'cat_encoders': encoders,
        'metrics': {
            'accuracy': rf_acc,
            'precision': rf_prec,
            'recall': rf_rec,
            'f1': rf_f1,
            'roc_auc': rf_auc
        }
    }
    artifact_path = os.path.join(models_dir, "staynest_cancellation_model.pkl")
    with open(artifact_path, 'wb') as f:
        pickle.dump(model_artifact, f)
    print(f"\n5. Successfully saved serialized trained model artifact: {artifact_path}")

    # Generate Feature Importance Chart
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    feat_imp = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(9, 6))
    sns.barplot(data=feat_imp.head(10), x='Importance', y='Feature', hue='Feature', palette='mako', legend=False)
    plt.title('Top 10 Predictors of Booking Cancellations (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Feature Importance Score', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "7_ml_feature_importance.png"), dpi=300)
    plt.close()
    print("Saved feature importance chart to reports/figures/7_ml_feature_importance.png")

if __name__ == "__main__":
    train_and_evaluate_models()
