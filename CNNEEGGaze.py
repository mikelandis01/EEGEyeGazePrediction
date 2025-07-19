import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, MaxPooling1D, Conv1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
import glob
import os

# EEG Channel labels (adjust as per your cap layout)
channel_labels = [f'Ch{idx+1}' for idx in range(128)]  # Placeholder for EEG channel labels

# Load and preprocess data
def load_and_preprocess(dataset_path):
    files = glob.glob(os.path.join(dataset_path, "*.mat"))
    X_data, y_gaze_x, y_gaze_y, y_pupil = [], [], [], []

    # First, find the shortest recording length across all files
    min_length = float('inf')
    for file in files:
        mat_data = scipy.io.loadmat(file)
        sEEG = mat_data['sEEG'][0,0]
        length = sEEG['data'].shape[1]
        if length < min_length:
            min_length = length

    print(f"DEBUG: Minimum length found across files: {min_length}")

    # Now, load the data, trimming to the minimum length
    for file in files:
        mat_data = scipy.io.loadmat(file)
        sEEG = mat_data['sEEG'][0,0]
        eeg = sEEG['data'][:128, :min_length].T

        gaze_x = np.clip(sEEG['data'][130, :min_length] / 800, 0, 1)
        gaze_y = np.clip(sEEG['data'][131, :min_length] / 600, 0, 1)
        pupil = np.log1p(sEEG['data'][132, :min_length])

        X_data.append(eeg)
        y_gaze_x.append(gaze_x)
        y_gaze_y.append(gaze_y)
        y_pupil.append(pupil)

    X_data = np.array(X_data, dtype=np.float32)
    y_gaze_x = np.array(y_gaze_x, dtype=np.float32)
    y_gaze_y = np.array(y_gaze_y, dtype=np.float32)
    y_pupil = np.array(y_pupil, dtype=np.float32)

    print("DEBUG: X_data shape before scaling:", X_data.shape)
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data.reshape(-1, X_data.shape[-1])).reshape(X_data.shape)

    pupil_scaler = MinMaxScaler()
    y_pupil = pupil_scaler.fit_transform(y_pupil.reshape(-1, 1)).reshape(y_pupil.shape)

    print("DEBUG: X_data shape after scaling:", X_data.shape)

    return X_data, y_gaze_x, y_gaze_y, y_pupil

# Segment data into sections
def segment_data(X, y_x, y_y, y_p, segment_size=50):  # You changed segment_size to 50
    segments_X, segments_yx, segments_yy, segments_yp = [], [], [], []
    for i in range(X.shape[0]):
        for start in range(0, X.shape[1] - segment_size + 1, segment_size):
            end = start + segment_size
            segments_X.append(X[i, start:end, :])
            segments_yx.append(y_x[i, start:end])
            segments_yy.append(y_y[i, start:end])
            segments_yp.append(y_p[i, start:end])

    print("DEBUG: segments_X shape:", np.array(segments_X).shape)

    return (np.array(segments_X), np.array(segments_yx), np.array(segments_yy), np.array(segments_yp))

# Model definition
def create_model(samples, channels):
    print(f"DEBUG: Model input shape: (samples={samples}, channels={channels})")
    inputs = Input(shape=(samples, channels))

    x = Conv1D(16, kernel_size=5, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)

    gaze_x = Dense(64, activation='relu')(x)
    gaze_x = Dense(samples, activation='sigmoid', name='gaze_x')(gaze_x)

    gaze_y = Dense(64, activation='relu')(x)
    gaze_y = Dense(samples, activation='sigmoid', name='gaze_y')(gaze_y)

    pupil = Dense(64, activation='relu')(x)
    pupil = Dense(samples, activation='linear')(pupil)
    pupil = Reshape((samples, 1), name='pupil_size')(pupil)

    model = Model(inputs, [gaze_x, gaze_y, pupil])

    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-5)

    model.compile(optimizer=optimizer,
                  loss={'gaze_x': 'mse', 'gaze_y': 'mse', 'pupil_size': 'huber'},
                  metrics={'gaze_x': 'mae', 'gaze_y': 'mae', 'pupil_size': 'mae'})
    return model

# Plot training curves
def plot_training_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Total Training Loss')
    plt.plot(history.history['val_loss'], label='Total Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs("visuals", exist_ok=True)
    plt.savefig("visuals/training_curves.png")
    plt.close()

# Plot sorted channel importance (top 20 only)
def plot_top_channel_importance(channel_importance):
    sorted_indices = np.argsort(channel_importance)[::-1][:20]
    sorted_importance = channel_importance[sorted_indices]
    sorted_labels = [f'Ch{idx+1}' for idx in sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_labels, sorted_importance)
    plt.xlabel('EEG Channel')
    plt.ylabel('Importance (Mean Abs Weight)')
    plt.title('Top 20 Channel Importance (First Conv Layer)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visuals/top_channel_importance.png")
    plt.close()

# Plot correlation and MAE trends
def plot_metrics_trend(corr_x_list, corr_y_list, mae_x_list, mae_y_list):
    epochs = np.arange(1, len(corr_x_list) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, corr_x_list, label='Gaze X Correlation')
    plt.plot(epochs, corr_y_list, label='Gaze Y Correlation')
    plt.title('Correlation Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs("visuals", exist_ok=True)
    plt.savefig("visuals/correlation_over_epochs.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mae_x_list, label='Gaze X MAE')
    plt.plot(epochs, mae_y_list, label='Gaze Y MAE')
    plt.title('MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("visuals/mae_over_epochs.png")
    plt.close()

# Main execution
dataset_path = "./dataset/"
X, y_gaze_x, y_gaze_y, y_pupil = load_and_preprocess(dataset_path)

print("DEBUG: Original X shape:", X.shape)

X_seg, yx_seg, yy_seg, yp_seg = segment_data(X, y_gaze_x, y_gaze_y, y_pupil, segment_size=50)

print("DEBUG: Segmented X shape:", X_seg.shape)

X_train, X_test, yx_train, yx_test, yy_train, yy_test, yp_train, yp_test = train_test_split(
    X_seg, yx_seg, yy_seg, yp_seg, test_size=0.2, random_state=42)

print("DEBUG: X_train shape before new axis:", X_train.shape)

X_train = X_train[..., np.newaxis].squeeze(-1)
X_test = X_test[..., np.newaxis].squeeze(-1)

print("DEBUG: X_train final shape:", X_train.shape)
print("DEBUG: X_test final shape:", X_test.shape)

total_samples = X_train.shape[1]
total_channels = X_train.shape[2]
print(f"DEBUG: channels from data: {total_channels}, samples: {total_samples}")

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

model = create_model(total_samples, total_channels)

corr_x_list, corr_y_list = [], []
mae_x_list, mae_y_list = []

epochs = 200
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    history = model.fit(X_train, {'gaze_x': yx_train, 'gaze_y': yy_train, 'pupil_size': yp_train},
                        validation_data=(X_test, {'gaze_x': yx_test, 'gaze_y': yy_test, 'pupil_size': yp_test}),
                        epochs=1,
                        batch_size=8,
                        callbacks=[lr_scheduler],
                        verbose=1)

    y_pred_gaze_x, y_pred_gaze_y, _ = model.predict(X_test)

    pred_x = y_pred_gaze_x.flatten()
    true_x = yx_test.flatten()

    pred_y = y_pred_gaze_y.flatten()
    true_y = yy_test.flatten()

    corr_x, _ = pearsonr(true_x, pred_x)
    corr_y, _ = pearsonr(true_y, pred_y)

    corr_x_list.append(corr_x)
    corr_y_list.append(corr_y)

    mae_x = np.mean(np.abs(true_x - pred_x))
    mae_y = np.mean(np.abs(true_y - pred_y))

    mae_x_list.append(mae_x)
    mae_y_list.append(mae_y)

    print(f"Epoch {epoch + 1}: Corr X: {corr_x:.3f}, Corr Y: {corr_y:.3f}, MAE X: {mae_x:.3f}, MAE Y: {mae_y:.3f}")

    current_lr = tf.keras.backend.get_value(model.optimizer.learning_rate)
    print(f"Learning Rate: {current_lr}")

plot_metrics_trend(corr_x_list, corr_y_list, mae_x_list, mae_y_list)

results = model.evaluate(X_test, {'gaze_x': yx_test, 'gaze_y': yy_test, 'pupil_size': yp_test})
print(f"Test Losses & Metrics: {results}")

first_conv = model.get_layer(index=1)
weights = first_conv.get_weights()[0]
print("DEBUG: Weights shape:", weights.shape)

abs_weights = np.abs(weights)
print("DEBUG: abs_weights shape:", abs_weights.shape)

channel_importance = np.mean(abs_weights, axis=(0, 2)).flatten()
print("DEBUG: channel_importance shape:", channel_importance.shape)

plot_top_channel_importance(channel_importance)

plt.figure(figsize=(12, 5))
plt.bar(np.arange(len(channel_importance)), channel_importance)
plt.xlabel('EEG Channel Index')
plt.ylabel('Importance (Mean Abs Weight)')
plt.title('Channel Importance (First Conv Layer)')
plt.tight_layout()
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/channel_importance.png")
plt.close()

os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/eeg_gaze_CNN_model.h5")
print("Model saved to 'saved_model/eeg_gaze_CNN_model.h5'")
