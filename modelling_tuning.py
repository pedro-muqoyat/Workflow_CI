import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DAGSHUB_USERNAME = "pedro-muqoyat"
DAGSHUB_REPO = "Eksperimen_SML_lifani"

# Membaca token rahasia yang diinjeksikan oleh GitHub Actions
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN")
if not DAGSHUB_TOKEN:
    raise ValueError("[GALAT FATAL] DAGSHUB_TOKEN tidak ditemukan di environment variables!")

os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

remote_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
mlflow.set_tracking_uri(remote_uri)
mlflow.set_experiment("Eksperimen_Prediksi_Status_Akademik")

def train_and_log_model():
    # Rute data bersifat relatif menyesuaikan hierarki repo GitHub
    data_path = "./data/data_bersih.csv"
    df = pd.read_csv(data_path)
    
    target_col = "Target"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    base_model = RandomForestClassifier(random_state=42)
    param_space = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'class_weight': ['balanced'] 
    }
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_space, cv=3, scoring='accuracy', n_jobs=-1)
    
    with mlflow.start_run(run_name="Run_CI_CD_Pipeline"):
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test_scaled)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("akurasi_pengujian", accuracy_score(y_test, predictions))
        mlflow.log_metric("f1_score_bobot", f1_score(y_test, predictions, average='weighted'))
        mlflow.sklearn.log_model(best_model, "model_klasifikasi_mahasiswa")
        
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 4))
        labels = ['Dropout', 'Enrolled', 'Graduate']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
        plt.title('Matriks Evaluasi Prediksi Model')
        plt.ylabel('Status Aktual')
        plt.xlabel('Hasil Prediksi')
        
        plt.savefig("evaluasi_confusion_matrix.png", bbox_inches='tight')
        mlflow.log_artifact("evaluasi_confusion_matrix.png")
        plt.close()

        # Ekspor arsitektur untuk dibungkus oleh Docker di langkah GitHub Actions berikutnya
        local_model_dir = "local_model_dir"
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
        mlflow.sklearn.save_model(best_model, local_model_dir)

if __name__ == "__main__":
    train_and_log_model()
