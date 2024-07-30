from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, PrecisionRecallDisplay, RocCurveDisplay, \
    confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, learning_curve
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    auc,
    roc_curve,
    ConfusionMatrixDisplay,
)
import warnings

warnings.filterwarnings("ignore")

path_dataset = "data/heart_preprocessed.csv"

def apprendimentoAlgoritmi():
    print("Inzio apprendimento supervisionato su dataset CSV \n")

    df = pd.read_csv(path_dataset)
    sc = StandardScaler()

    X = df.drop(columns='HeartDisease')
    sc.fit(X)
    X = sc.transform(X)
    y = df['HeartDisease']

    models = {
        'Logistic Regression': (LogisticRegression(), {'C': [0.001, 0.01, 0.1, 1, 10, 100]}),
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]}),
        'RandomForestClassifier': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}),
        'GrandientBoostingClassifier': (GradientBoostingClassifier(), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]}),
        'AdaBoostClassifier': (AdaBoostClassifier(), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]})
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Inizio Task 1: Classificazione")
    print(" Eseguo Grid Search con Kfold-cross-validation")

    best_models = {}

    for model_name, (model, params) in models.items():
        print(" Valutazione modello", model_name)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        best_model = grid_search.best_estimator_

        test_accuracy = best_model.score(X_test, y_test)
        y_pred = best_model.predict(X_test)
        best_models[model_name] = {'model': best_model, 'best_params': best_params, 'test_accuracy': test_accuracy,
                                   'report': classification_report(y_test, y_pred)}

        cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
        disp.plot()
        plt.savefig("img/" + model_name + 'ConfusionMatrix.png')
        plt.show()

        RocCurveDisplay.from_estimator(best_model, X_test, y_test)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.savefig("img/" + model_name + 'RocCurve.png')
        plt.show()

        PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test)
        plt.savefig("img/" + model_name + 'PrecisionRecallCurve.png')
        plt.show()

    for model_name, info in best_models.items():
        print(f"Modello: {model_name}")
        print(f"Migliori parametri: {info['best_params']}")
        print(f"Accuratezza sul set di test: {info['test_accuracy']:.2f}")
        print(info['report'])
        print("=" * 50)

    print("Fine apprendimento supervisionato su dataset CSV, metriche salvate con successo \n")

def apprendimentoCNN():
    print("Inzio apprendimento CNN su dataset immagini \n")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    path_immagini_melanoma = "data/train/train"
    batch_size_value = 256
    batch_img_size = 128

    data = tf.keras.utils.image_dataset_from_directory(
        path_immagini_melanoma,
        batch_size=batch_size_value,
        color_mode="rgb",
        image_size=(batch_img_size, batch_img_size),
        interpolation="bilinear",
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
    )

    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()

    data = data.map(lambda x, y: (x / 255, y))
    data.as_numpy_iterator().next()

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    test_size = int(len(data) * 0.1)

    print("Data_size:", len(data))
    print("Data_type:", type(data))
    print("train_size:", train_size)
    print("test_size:", test_size)
    print("val_size:", val_size)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)


    model = Sequential()
    model.add(
        Conv2D(
            16,
            (3, 3),
            1,
            activation="relu",
            input_shape=(batch_img_size, batch_img_size, 3),
        )
    )
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation="relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), 1, activation="relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

    print(model.summary())

    logdir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    hist = model.fit(
        train, epochs=20, validation_data=val, callbacks=[tensorboard_callback]
    )

    fig = plt.figure()
    plt.plot(hist.history["loss"], color="teal", label="loss")
    plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
    fig.suptitle("Loss", fontsize=20)
    plt.legend(loc="upper left")
    plt.savefig("img/LossCNN.png")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history["accuracy"], color="teal", label="accuracy")
    plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")
    fig.suptitle("Accuracy", fontsize=20)
    plt.legend(loc="upper left")
    plt.savefig("img/AccuracyCNN.png")
    plt.show()

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    y_true = []
    y_pred = []

    model.save(os.path.join("model", "melanomaCNNClassifier.h5"))

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
        y_true.extend(y)
        y_pred.extend(yhat)

    print("Precision: ", round(pre.result().numpy(), 2))
    print("Recall: ", round(re.result().numpy(), 2))
    print("Accuracy: ", round(acc.result().numpy(), 2))

    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    print(classification_report(y_true, y_pred_binary))

    # Confusion Matrix
    cm = confusion_matrix(y_true, np.round(y_pred))
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig("img/ConfusionMatrixCNN.png")
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig("img/RocCNN.png")
    plt.show()

    print("Fine apprendimento su dataset immagini, metriche salvate con successo \n")

if __name__ == '__main__':
    apprendimentoAlgoritmi()