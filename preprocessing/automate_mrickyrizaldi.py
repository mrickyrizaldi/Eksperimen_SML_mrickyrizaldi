from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from joblib import dump
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Path lokasi script dan direktori root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def log_transform(X):
    '''Fungsi untuk melakukan log transformasi pada data numerik tertentu'''
    return np.log1p(np.maximum(X, 0))


def build_preprocessing_pipeline(num_skewed_features, num_normal_features, cat_features):
    '''
    Fungsi membangun pipeline preprocessing untuk fitur numerik dan kategoris
    '''
    # Pipeline untuk fitur numerik yang terdistorsi
    num_skewed_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('transformer', FunctionTransformer(log_transform)),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline untuk fitur numerik yang normal
    num_normal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline untuk fitur kategoris
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan semua pipeline ke dalam ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num_skewed', num_skewed_pipeline, num_skewed_features),
        ('num_normal', num_normal_pipeline, num_normal_features),
        ('cat', cat_pipeline, cat_features)
    ])

    return preprocessor


def preprocess_pipeline(data, target_column, save_pipeline_path):
    '''
    Fungsi untuk melakukan preprocessing data dan menyimpan pipeline
    '''
    # Drop feature Pipe_Size_mm jika ada
    if 'Pipe_Size_mm' in data.columns:
        data = data.drop('Pipe_Size_mm', axis=1)

    # Pisahkan fitur numerik dan kategorikal
    num_features = data.select_dtypes(include=['number']).columns.tolist()
    cat_features = data.select_dtypes(include=['object']).columns.tolist()

    # Hilangkan target dari daftar kolom
    if target_column in num_features:
        num_features.remove(target_column)
    if target_column in cat_features:
        cat_features.remove(target_column)

    # Definisikan kolom yang ingin di-log transform
    num_skewed_features = [col for col in ['Thickness_mm', 'Material_Loss_Percent'] if col in num_features]
    num_normal_features = [col for col in num_features if col not in num_skewed_features]

    # Pisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Label Encoding untuk target
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Simpan encoder
    dump(label_encoder, save_pipeline_path.replace('.joblib', '_label_encoder.joblib'))

    # Bangun pipeline
    preprocessor = build_preprocessing_pipeline(num_skewed_features, num_normal_features, cat_features)

    # Fit-transform ke X_train dan transform X_test
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Simpan pipeline
    dump(preprocessor, save_pipeline_path)

    # Mendapatkan nama fitur hasil transformasi
    feature_column = []
    feature_column.extend(num_skewed_features)
    feature_column.extend(num_normal_features)
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(cat_features)
    feature_column.extend(cat_feature_names)

    return X_train, X_test, y_train, y_test, feature_column, target_column


if __name__ == "__main__":
    # Path input dataset
    dataset_path = os.path.join(ROOT_DIR, 'dataset_raw', 'market_pipe_thickness_loss_dataset.csv')
    data = pd.read_csv(dataset_path)

    # Buat folder output di preprocessing
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join(SCRIPT_DIR, f'preprocessed_data_auto_{timestamp}')
    os.makedirs(output_folder, exist_ok=True)

    # Jalankan preprocessing
    X_train, X_test, y_train, y_test, feature_column, target_column = preprocess_pipeline(
        data=data,
        target_column='Condition',
        save_pipeline_path=os.path.join(output_folder, 'pipeline.joblib')
    )

    # Simpan output ke CSV
    pd.DataFrame(X_train, columns=feature_column).to_csv(os.path.join(output_folder, 'X_train.csv'), index=False)
    pd.DataFrame(X_test, columns=feature_column).to_csv(os.path.join(output_folder, 'X_test.csv'), index=False)
    pd.DataFrame(y_train, columns=[target_column]).to_csv(os.path.join(output_folder, 'y_train.csv'), index=False)
    pd.DataFrame(y_test, columns=[target_column]).to_csv(os.path.join(output_folder, 'y_test.csv'), index=False)
    pd.Series(feature_column).to_csv(os.path.join(output_folder, 'feature_names.csv'), index=False, header=False)

    print(f"Hasil preprocessing disimpan di: {output_folder}")