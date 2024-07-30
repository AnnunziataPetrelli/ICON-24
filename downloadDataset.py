import os
import kaggle

dataset_name2 = "rahimanshu/cardiomegaly-disease-prediction-using-cnn"
dataset_name = "fedesoriano/heart-failure-prediction"
path_data_raw = "data/"


def download_kaggle_dataset(path_dataset_name, path_data_raw):
    kaggle.api.authenticate()
    train_path = os.path.join(path_data_raw, "train")
    test_path = os.path.join(path_data_raw, "test")
    dataset_path = os.path.join(path_data_raw, "heart.csv")
    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(dataset_path):
        print("I dataset sono stati già scaricati")
        return 1
    if path_dataset_name == dataset_name2 and os.path.exists(train_path) and os.path.exists(test_path):
        return 1
    if path_dataset_name == dataset_name and os.path.exists(dataset_path):
        return 1
    else:
        try:
            kaggle.api.dataset_download_files(
                path_dataset_name, path=path_data_raw, unzip=True, force=True
            )
            print("Dataset scaricato ed estratto correttamente.")
        except Exception as e:
            print(f"Si è verificato un errore durante il download del dataset: {e}")
            return 1
    return 0

