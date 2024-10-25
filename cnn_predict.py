import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import time


class AdvancedPreprocessing:
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    def remove_outliers(self, data, z_threshold=3):
        z_scores = np.abs(stats.zscore(data))
        return data[np.all(z_scores < z_threshold, axis=1)]

    def feature_engineering(self, data):
        data['mean'] = data.mean(axis=1)
        data['std'] = data.std(axis=1)
        data['skew'] = stats.skew(data, axis=1)
        data['kurtosis'] = stats.kurtosis(data, axis=1)

        for i in range(min(5, data.shape[1])):
            for j in range(i + 1, min(6, data.shape[1])):
                data[f'interact_{i}_{j}'] = data.iloc[:, i] * data.iloc[:, j]

        return data


class ComplexCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.training_time = 0
        self.history = None

    def residual_block(self, x, filters, kernel_size=3):
        shortcut = x
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv1D(filters, 1, padding='same')(shortcut)

        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.ReLU()(x)
        return x

    def squeeze_excitation_block(self, x, ratio=16):
        filters = x.shape[-1]
        se = tf.keras.layers.GlobalAveragePooling1D()(x)
        se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
        se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
        se = tf.keras.layers.Reshape((1, filters))(se)
        return tf.keras.layers.multiply([x, se])

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv1D(64, 7, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)

        x = self.residual_block(x, 64)
        x = self.squeeze_excitation_block(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = self.residual_block(x, 128)
        x = self.squeeze_excitation_block(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = self.residual_block(x, 256)
        x = self.squeeze_excitation_block(x)

        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.concatenate([avg_pool, max_pool])

        dense1 = tf.keras.layers.Dense(512, activation='relu')(x)
        dense2 = tf.keras.layers.Dense(512, activation='relu')(dense1)
        dense2 = tf.keras.layers.Add()([dense1, dense2])

        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dense2)

        return tf.keras.Model(inputs, outputs)

    def custom_learning_rate_schedule(self, epoch):
        initial_lr = 0.001
        warmup_epochs = 5
        decay_rate = 0.1

        if epoch < warmup_epochs:
            return initial_lr * ((epoch + 1) / warmup_epochs)
        else:
            return initial_lr * (decay_rate ** ((epoch - warmup_epochs) // 10))

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.custom_learning_rate_schedule)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        start_time = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr_scheduler, early_stopping, reduce_lr],
            verbose=1
        )
        self.training_time = time.time() - start_time

    def evaluate_model(self, X_test, y_test):
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'training_time': self.training_time
        }


def main():
    preprocessing = AdvancedPreprocessing()

    data = pd.read_csv('data_predict_cnn.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    X = preprocessing.remove_outliers(X)
    X = preprocessing.feature_engineering(X)
    X = preprocessing.standard_scaler.fit_transform(X)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = ComplexCNN(input_shape=(X.shape[1], 1), num_classes=len(np.unique(y)))

        model.train(X_train, y_train, X_val, y_val)
        results = model.evaluate_model(X_val, y_val)
        cv_results.append(results)

    avg_accuracy = np.mean([r['test_accuracy'] for r in cv_results])
    avg_precision = np.mean([r['test_precision'] for r in cv_results])
    avg_recall = np.mean([r['test_recall'] for r in cv_results])

    print("\nAverage Cross-Validation Results:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")


if __name__ == "__main__":
    main()
