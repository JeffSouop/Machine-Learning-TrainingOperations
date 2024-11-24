import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_image_paths(file_path):
    """Charge les chemins des images depuis un fichier (une ligne par chemin)."""
    with open(file_path, 'r') as f:
        paths = f.read().splitlines()
    return paths

def load_images_and_labels(image_paths, labels=None):
    """Charge les images depuis les chemins et retourne les images et leurs labels."""
    images = []
    loaded_labels = [] if labels is not None else None

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image non trouvée : {img_path}")
            continue
        img = cv2.resize(img, (112, 112))
        img = img / 255.0
        images.append(img)
        
        if labels is not None:
            loaded_labels.append(labels[i])

    images = np.array(images)
    if labels is not None:
        loaded_labels = np.array(loaded_labels)
        return images, loaded_labels
    return images

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

print("Chargement des données d'entraînement...")
train_paths_file = 'train_100K.csv'
train_labels_file = 'train_labels.csv'

train_paths = load_image_paths(train_paths_file)
with open(train_labels_file, 'r') as f:
    train_labels = [float(line.strip()) for line in f]

X, y = load_images_and_labels(train_paths, train_labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Construction et entraînement du modèle...")
model = build_cnn_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=10
)

model.save('model_cnn_face_occlusion.h5')

print("Chargement des données de test...")
test_paths_file = 'test_students.csv'
test_paths = load_image_paths(test_paths_file)

X_test = load_images_and_labels(test_paths)

print("Génération des prédictions...")
test_predictions = model.predict(X_test)

output_file = 'predictions_SOUOP.txt'
with open(output_file, 'w') as f:
    for img_path, pred in zip(test_paths, test_predictions):
        image_name = os.path.basename(img_path)
        f.write(f"{image_name}.{pred[0]:.2f}\n")

print(f"Prédictions enregistrées dans {output_file}")

plt.figure()
plt.plot(history.history['loss'], label='Loss (Entraînement)')
plt.plot(history.history['val_loss'], label='Loss (Validation)')
plt.title('Courbe de perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['mae'], label='MAE (Entraînement)')
plt.plot(history.history['val_mae'], label='MAE (Validation)')
plt.title('Courbe MAE')
plt.xlabel('Époque')
plt.ylabel('MAE')
plt.legend()
plt.show()