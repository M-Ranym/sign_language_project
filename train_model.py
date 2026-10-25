import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import random
import matplotlib.pyplot as plt

ACTIONS = ['blanc', 'hello', 'goodbye', 'yes', 'no', 'thank you', 'i love you']
SEQUENCE_LENGTH = 15
DATA_PATH = './gesture_data'
NUM_SEQUENCES = 50

def augment_sequence(sequence):
    augmented = np.copy(sequence)
    if not np.all(augmented[:, 132:258] == 0):
        angle = random.uniform(-60, 60) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        scale = random.uniform(0.7, 1.3)
        trans_x, trans_y = random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)
        noise = np.random.normal(0, 0.06, 126)
        
        for t in range(SEQUENCE_LENGTH):
            hand_points = augmented[t, 132:258].reshape(-1, 3)[:, :2]
            hand_points = hand_points @ rotation
            hand_points *= scale
            hand_points[:, 0] += trans_x
            hand_points[:, 1] += trans_y
            augmented[t, 132:258:3] = hand_points[:, 0]
            augmented[t, 133:258:3] = hand_points[:, 1]
            augmented[t, 132:258] += noise
    return augmented

X, y = [], []
for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    sequences = []
    for sequence in range(NUM_SEQUENCES):
        seq_path = os.path.join(action_path, f'seq_{sequence}.npy')
        if os.path.exists(seq_path):
            sequence_data = np.load(seq_path)
            sequences.append(sequence_data)
        else:
            print(f"Missing: {seq_path}")
    if len(sequences) > 0:
        while len(sequences) < NUM_SEQUENCES:
            sequences.append(random.choice(sequences))
        sequences = sequences[:NUM_SEQUENCES]
        X.extend(sequences)
        y.extend([action] * NUM_SEQUENCES)

X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

class_weights = len(y_encoded) / (len(ACTIONS) * np.bincount(y_encoded))
class_weights = {i: weight for i, weight in enumerate(class_weights)}

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

X_train_augmented = np.array([augment_sequence(seq) for seq in X_train])

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(SEQUENCE_LENGTH, 292)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(len(ACTIONS), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

history = model.fit(X_train_augmented, y_train, 
                    epochs=300,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    class_weight=class_weights)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.close()

model.save('gesture_model.keras')
np.save('label_encoder_classes.npy', label_encoder.classes_)