import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import math as math
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneebow.rotor import Rotor
def plot_anomalies(text, X_test, res, threshold, log=False):
    plt.figure(figsize=(6, 6))
    n = min(X_test.shape[0], res.shape[0])

    x = np.arange(0, n)
    t_line = np.zeros(n)
    if (log):
        thresh = np.log(threshold)
    else:
        thresh = threshold
    t_line[:] = thresh

    c = 0
    for p in range(0, n):
        re = (np.linalg.norm(X_test[p, ...] - res[p, ...], 2))
        if (log):
            re = np.log(re)
        if (re >= thresh):
            c = c + 1
            plt.scatter(p, re, c='r')
        else:
            plt.scatter(p, re, c='g')
    plt.plot(x, t_line, 'b')
    plt.ylabel('Reconstruction error')
    plt.xlabel('Samples')
    plt.savefig('Img/AutoencoderAnomaly' + text + '.png')
    plt.show()


def eval_error(X_test, res_test, std=False):
    n = min(X_test.shape[0], res_test.shape[0])
    n_features = X_test.shape[1]  # Numero di feature

    overall_mean = 0
    dict_means = {}
    dict_std = {} if std else None

    for feature_index in range(n_features):
        feature_errors = []

        for sample_index in range(n):
            re = np.linalg.norm(X_test[sample_index] - res_test[sample_index], 2)
            feature_errors.append(re)

        mean_error = np.mean(feature_errors)
        overall_mean += mean_error
        dict_means[feature_index] = np.round(mean_error, 4)

        if std:
            std_error = np.std(feature_errors)
            dict_std[feature_index] = np.round(std_error, 4)

    overall_mean /= n_features

    if std:
        return overall_mean, dict_means, dict_std
    else:
        return overall_mean, dict_means

def select_threshold(X, res, mean_rec_err, perc=.98, nit=10000, step=None, out=False):
    """
    INPUT

    - X : training data of the autoencoder
    - res: reconstructed data from autoencoder
    - mean_rec_err:  mean reconstruction error obtained on training data
    - perc: percentage of training data for which the reconstruction error must be below the mean_rec_err multiplied by a factor
    - nit: max number of iterations
    - step: step with which increase the multiplication factor at every step

    OUTPUT:

    - threshold: threshold for reconstruction error on test data, for which considering a sample anomalous

    NB: An higher percentage value will increase the threshold, resulting in a looser anomaly detection.

    """

    counter = 0
    target = 0
    dim = X.shape[0]
    if (step is None or step >= 1. or step <= 1e-5):
        step = 1 - perc  # step estimation from perc for a better accuracy in factor estimation

    if (perc >= 1. or perc <= 1e-2):
        return mean_rec_err
    else:
        factor = 1 - step  # This subtraction is needed for the factor to be equal to one at the first step of while cycle

    while (target < perc and counter < nit):
        j = 0
        counter = counter + 1
        factor = factor + step

        for n in range(dim):
            err = np.linalg.norm(X[n, ...] - res[n, ...], 2)
            if (abs(err <= mean_rec_err * factor)):
                j = j + 1
        target = np.round(j / dim, 2)
        if (out):
            print(target, factor)

    threshold = np.round(factor * mean_rec_err, 3)
    return threshold

def detect_anomalies(X_test, res_test, threshold, out=True):
    """
    INPUT

    - X_test : test samples
    - res_test: reconstructed test samples
    - threshold: threshold for reconstruction error on test data, for which considering a sample anomalous. See select_threshold for details

    OUTPUT:

    - idx_anomalies: list of indices of test samples deemed anomalous on the basis of the given threshold.


    """

    dim = X_test.shape[0]
    idx_anomalies = []

    for i in range(dim):
        err = np.linalg.norm(X_test[i, ...] - res_test[i, ...], 2)
        if (abs(err > threshold)):
            idx_anomalies.append(i)

    if (out):
        print('\nPercentage of anomalies in tested data:' + str(np.round(len(idx_anomalies) / X_test.shape[0], 2)))

    return idx_anomalies

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def autoencoderAnomalyDetection():
    print("Inizio apprendimento non supervisionato con autoencoder \n")

    data = pd.read_csv("data/heart_preprocessed.csv")
    set_all_seeds(42)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X_train, X_test = train_test_split(scaled_data, test_size=0.2)

    input_dim = X_train.shape[1]
    encoding_dim = 128
    hidden_dim = 256

    input_data = tf.keras.layers.Input(shape=(input_dim,))
    hidden1 = tf.keras.layers.Dense(hidden_dim*2, activation='relu')(input_data)
    encoder = tf.keras.layers.Dense(encoding_dim*2, activation='relu')(hidden1)
    hidden2 = tf.keras.layers.Dense(hidden_dim*2, activation='relu')(encoder)
    decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(hidden2)

    autoencoder = tf.keras.models.Model(inputs=input_data, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, validation_split=0.2)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper right')
    plt.savefig('img/LossAutoencoder.png')
    plt.show()

    print("Effettuo la predizione sul set di train")
    predictions = autoencoder.predict(X_train)
    predictionsTrain = predictions
    mse_train = round(mean_squared_error(X_train, predictions), 5)
    mae_train = round(mean_absolute_error(X_train, predictions), 5)
    print("Errore quadratico medio (MSE) train:", mse_train)
    print("Errore assoluto medio (MAE) train:", mae_train)
    print("R2:", r2_score(X_train, predictions))

    print("Effettuo la predizione sul set di test")
    predictions = autoencoder.predict(X_test)
    predictionsTest = predictions
    mse_test = round(mean_squared_error(X_test, predictions), 5)
    mae_test = round(mean_absolute_error(X_test, predictions), 5)
    print("Errore quadratico medio (MSE) test:", mse_test)
    print("Errore assoluto medio (MAE) test:", mae_test)
    print("R2:", r2_score(X_test, predictions))

    threshold = select_threshold(X_train, predictionsTrain, mae_train, perc=.98)
    idx_anomalies_train = detect_anomalies(X_train, predictionsTrain, threshold)
    idx_anomalies_test = detect_anomalies(X_test, predictionsTest, threshold)
    plot_anomalies("Train", X_train, predictionsTrain, threshold)
    plot_anomalies("Test", X_test, predictionsTest, threshold)

    mae_train, dict_mae_train = eval_error(X_train, predictionsTrain)
    mae_test, dict_mae_test = eval_error(X_test, predictionsTest)

    df_anomalies_train = data.iloc[idx_anomalies_train]
    df_anomalies_test = data.iloc[idx_anomalies_test]

    df_anomalies = pd.concat([df_anomalies_train, df_anomalies_test])

    df_anomalies.to_csv('data/anomalieAutoencoder.csv', index=False)

    print("Anomalie trovate e salvate con successo! \n")

def KMeansAnomalyDetection():
    print("Inizio apprendimento non supervisionato con KMeans \n")

    df = pd.read_csv("data/heart_preprocessed.csv")
    X = df

    inertia_values = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(10, 10))
    plt.plot(range(1, 11), inertia_values, marker='o')
    plt.title("Analisi del gomito")
    plt.xlabel("Numero di cluster")
    plt.ylabel("Inertia")
    plt.xticks(range(1, 11))
    plt.savefig("Img/Analisi_del_gomito.png")
    plt.show()

    silhouette_scores = {}
    for k in range(2, 25, 1):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(X)
        labels_k = kmeans.labels_
        score_k = metrics.silhouette_score(X, labels_k)
        silhouette_scores[k] = score_k
        print("Testo kMeans con k = %d\tSS: %5.4f" % (k, score_k))

    plt.figure(figsize=(16, 5))
    plt.plot(silhouette_scores.values())
    plt.xticks(range(0, 23, 1), silhouette_scores.keys())
    plt.title("Silhouette Metric")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.axvline(1, color="r")
    plt.savefig("img/Silhouette.png")
    plt.show()

    print("Addestro kmeans per il suo k migliore ")

    kmeans = KMeans(n_clusters=3, random_state=42)

    kmeans.fit_predict(X)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    distances = np.linalg.norm(X - cluster_centers[cluster_labels], axis=1)

    anomaly_thresholt = np.percentile(distances, 95)
    anomalies = X[distances > anomaly_thresholt]

    df_merged = pd.merge(df[:], anomalies, left_index=True, right_index=True, how='inner')
    df_merged.to_csv("data/anomalieKMeans.csv")

    print("Anomalie trovate e salvate con successo! \n")

def calculate_kn_distance(X, neigh=2):
    neigh = NearestNeighbors(n_neighbors=neigh)
    nbrs = neigh.fit (X)
    distances, indices = nbrs.kneighbors(X)
    return distances [:,1:]. reshape (-1)


def get_eps(X, neigh=2) :
    eps_dist = np.sort(calculate_kn_distance(X, neigh=neigh))
    rotor = Rotor()
    curve_xy = np.concatenate([np.arange(eps_dist.shape[0]).reshape(-1, 1), eps_dist.reshape (-1, 1)],1)
    rotor.fit_rotate(curve_xy)
    rotor.plot_elbow()
    e_idx = rotor.get_elbow_index()
    return curve_xy[e_idx]

def DBScan():
    print("Inizio apprendimento non supervisionato con DBScan \n")

    df = pd.read_csv("data/heart_preprocessed.csv")
    idx, eps = get_eps(df)
    plt.savefig("img/epsDBScan.png")
    plt.show()

    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(df)

    df['Cluster_DBscan'] = clusters

    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"Numero di cluster ottenuti: {num_clusters}")

    anomalies = df[df['Cluster_DBscan'] == -1]
    anomalies.to_csv("data/anomalieDBScan.csv", index=False)

    print("Anomalie trovate e salvate con successo! \n")

