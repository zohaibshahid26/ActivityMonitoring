from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

app = Flask(__name__)

data_path = "D:\\5th Semester\\ML\\ML Project\\bbh\\"
def load_dataset(data_path):
    # Load training data and training labels from files
    train_accel_ms = np.load(data_path + "training/trainMSAccelerometer.npy")
    train_gyro_ms = np.load(data_path + "training/trainMSGyroscope.npy")
    train_accel = np.load(data_path + "training/trainAccelerometer.npy")
    train_gravity = np.load(data_path + "training/trainGravity.npy")
    train_accel_jin = np.load(data_path + "training/trainJinsAccelerometer.npy")
    train_gyro_jin = np.load(data_path + "training/trainJinsGyroscope.npy")
    train_lin_accel = np.load(data_path + "training/trainLinearAcceleration.npy")
    train_magnetometer = np.load(data_path + "training/trainMagnetometer.npy")
    train_gyro = np.load(data_path + "training/trainGyroscope.npy")
    train_labels = np.load(data_path + "training/trainlabels.npy")

    # Load testing data and testing labels from files
    test_accel_ms = np.load(data_path + "testing/testMSAccelerometer.npy")
    test_gyro_ms = np.load(data_path + "testing/testMSGyroscope.npy")
    test_accel = np.load(data_path + "testing/testAccelerometer.npy")
    test_gravity = np.load(data_path + "testing/testGravity.npy")
    test_accel_jin = np.load(data_path + "testing/testJinsAccelerometer.npy")
    test_gyro_jin = np.load(data_path + "testing/testJinsGyroscope.npy")
    test_lin_accel = np.load(data_path + "testing/testLinearAcceleration.npy")
    test_magnetometer = np.load(data_path + "testing/testMagnetometer.npy")
    test_gyro = np.load(data_path + "testing/testGyroscope.npy")
    test_labels = np.load(data_path + "testing/testlabels.npy")

    training_data = [train_accel_ms, train_gyro_ms, train_accel, train_gravity, train_accel_jin, train_gyro_jin,
                     train_lin_accel, train_magnetometer, train_gyro]
    testing_data = [test_accel_ms, test_gyro_ms, test_accel, test_gravity, test_accel_jin, test_gyro_jin,
                    test_lin_accel, test_magnetometer, test_gyro]

    return training_data, train_labels, testing_data, test_labels


def pre_process_dataset_scalar(data_sets):
    pre_processed_data = []
    for data_set in data_sets:
        scalar = StandardScaler()
        impute = SimpleImputer()
        N, T, S = data_set.shape
        reshaped_data = np.reshape(data_set, (N, T * S))
        imputed_data = impute.fit_transform(reshaped_data)
        pre_processed_data.append(scalar.fit_transform(imputed_data).reshape((N, T, S)))
    return pre_processed_data


def pre_process_dataset_normalizer(data_sets):
    pre_processed_data = []
    for data_set in data_sets:
        normalizer = Normalizer()
        impute = SimpleImputer()
        N, T, S = data_set.shape
        reshaped_data = np.reshape(data_set, (N, T * S))
        imputed_data = impute.fit_transform(reshaped_data)
        pre_processed_data.append(normalizer.fit_transform(imputed_data).reshape((N, T, S)))
    return pre_processed_data


def extract_features(data):
    features = []
    for data_set in data:
        features.append(np.mean(data_set, axis=1))
        features.append(np.max(data_set, axis=1))
        features.append(np.min(data_set, axis=1))
        features.append(np.std(data_set, axis=1))
        features.append(np.var(data_set, axis=1))
        features.append(np.median(data_set, axis=1))
        features.append(np.percentile(data_set, 25, axis=1))
        features.append(np.percentile(data_set, 75, axis=1))
    return np.concatenate(features, axis=1)


def RandomForest(X_train_normalizer, train_labels, test_labels, X_test_normalizer):
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train_normalizer, train_labels)
    predictions = random_forest.predict(X_test_normalizer)
    accuracy = accuracy_score(test_labels, predictions)
    averageF1 = f1_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    rf_confusion = confusion_matrix(test_labels, predictions)
    evaluation_metrics = {
        'Accuracy': float(accuracy),
        'F1_Score': float(averageF1),
        'Recall_Score': float(recall),
        'Confusion_Matrix': rf_confusion
    }
    return evaluation_metrics


def SupportVectorMachine(X_train_normalizer, train_labels, test_labels, X_test_normalizer):
    SVM = SVC()
    SVM.fit(X_train_normalizer, train_labels)
    predicted_labels = SVM.predict(X_test_normalizer)
    accuracy: float = accuracy_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels, average='macro')
    averageF1 = f1_score(test_labels, predicted_labels, average='macro')
    SVM_confusion_matrix = confusion_matrix(test_labels, predicted_labels)
    evaluation_metrics = {
        'Accuracy': float(accuracy),
        'F1_Score': float(averageF1),
        'Recall_Score': float(recall),
        'Confusion_Matrix': SVM_confusion_matrix
    }
    return evaluation_metrics


def LRegression(X_train_scalar, train_labels, test_labels, X_test_scalar):
    Logistic_Regression = LogisticRegression(max_iter=3000)
    Logistic_Regression.fit(X_train_scalar, train_labels)
    predicted_labels = Logistic_Regression.predict(X_test_scalar)
    accuracy = accuracy_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels, average='macro')
    averageF1 = f1_score(test_labels, predicted_labels, average='macro')
    logistic_Regression_confusion_matrix = confusion_matrix(test_labels, predicted_labels)
    evaluation_metrics = {
        'Accuracy': float(accuracy),
        'F1_Score': float(averageF1),
        'Recall_Score': float(recall),
        'Confusion_Matrix': logistic_Regression_confusion_matrix
    }
    return evaluation_metrics


def NaiveBayes(X_train_scalar, train_labels, test_labels, X_test_scalar):
    Naive_Bayes = GaussianNB()
    Naive_Bayes.fit(X_train_scalar, train_labels)
    predicted_labels = Naive_Bayes.predict(X_test_scalar)
    accuracy = accuracy_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels, average='macro')
    averageF1 = f1_score(test_labels, predicted_labels, average='macro')
    naive_bayes_confusion_matrix = confusion_matrix(test_labels, predicted_labels)
    evaluation_metrics = {
        'Accuracy': float(accuracy),
        'F1_Score': float(averageF1),
        'Recall_Score': float(recall),
        'Confusion_Matrix': naive_bayes_confusion_matrix
    }
    return evaluation_metrics


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        selected_algorithm = request.form.get('algorithm')
        training_data, train_labels, testing_data, test_labels = load_dataset(data_path)

        X_train_normalizer = extract_features(pre_process_dataset_normalizer(training_data))
        X_test_normalizer = extract_features(pre_process_dataset_normalizer(testing_data))

        X_train_scalar = extract_features(pre_process_dataset_scalar(training_data))
        X_test_scalar = extract_features(pre_process_dataset_scalar(testing_data))

        if selected_algorithm == "Random Forest Classifier":
            evaluation_metrics = RandomForest(X_train_normalizer, train_labels, test_labels, X_test_normalizer)
        elif selected_algorithm == "Support Vector Machine":
            evaluation_metrics = SupportVectorMachine(X_train_normalizer, train_labels, test_labels, X_test_normalizer)
        elif selected_algorithm == "Logistic Regression":
            evaluation_metrics = LRegression(X_train_scalar, train_labels, test_labels, X_test_scalar)
        elif selected_algorithm == "Naive Bayes":
            evaluation_metrics = NaiveBayes(X_train_scalar, train_labels, test_labels, X_test_scalar)
        else:
            evaluation_metrics = {}
    else:
        selected_algorithm = "No Algorithm Selected"
        evaluation_metrics ={
            "Accuracy": 0,
            "F1_Score": 0,
            "Recall_Score": 0,
            "Confusion_Matrix": []
        }
    return render_template('results.html', algorithm=selected_algorithm, evaluation_metrics=evaluation_metrics)


if __name__ == "__main__":
    app.run(debug=True)
