import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_data
import os

# --- Constantes y Configuración ---
DATASET_FILE = 'pong_q_expert_dataset.npy'
MODEL_SAVE_PATH = 'pong_bc_q_expert_model.h5'
NUM_ACTIONS = 3 # Subir, Bajar, Quieto (0, 1, 2)

# Hiperparámetros de entrenamiento
EPOCHS = 1000
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42 # Para reproducibilidad en shuffle y train_test_split

# --- Carga y Preparación del Dataset ---
print(f"Cargando dataset desde: {DATASET_FILE}")
dataset = np.load(DATASET_FILE, allow_pickle=True)

if dataset.size == 0:
    print("Error: El dataset está vacío. No se puede entrenar.")
    exit()

# Extraer estados (features) y acciones (labels)
states = np.array([item[0] for item in dataset])
actions = np.array([item[1] for item in dataset])
print(f"Total de muestras cargadas inicialmente: {len(states)}")

if len(states) == 0:
    print("Error: El dataset cargado no contiene muestras.")
    exit()

# --- 2. Shuffle de datos ---
print("\nBarajando los datos...")
states, actions = shuffle_data(states, actions, random_state=RANDOM_STATE)
print("Datos barajados.")

# Verificar el número de características de entrada (después de filtrar y antes de entrenar)
num_features = states.shape[1]
print(f"Número de características de entrada detectadas: {num_features}")
print(f"Forma de los estados (X) después de procesar: {states.shape}")
print(f"Forma de las acciones (y) después de procesar: {actions.shape}")

# --- 3. One-Hot Encoding de las Acciones ---
print(f"\nConvirtiendo acciones a formato one-hot (para {NUM_ACTIONS} clases)...")
actions_one_hot = to_categorical(actions, num_classes=NUM_ACTIONS)
if len(actions_one_hot) > 0:
    print(f"Forma de las acciones one-hot: {actions_one_hot.shape}")
else:
    print("Error: No se pudieron convertir las acciones a one-hot (dataset vacío después de procesar).")
    exit()

# --- 4. Dividir datos en conjuntos de entrenamiento y validación ---
print(f"\nDividiendo datos en entrenamiento y validación (split: {VALIDATION_SPLIT})...")
X_train, X_val, y_train, y_val = train_test_split(
    states, actions_one_hot,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_STATE, # Usar el mismo random_state para consistencia
    stratify=actions # Estratificar basado en las acciones originales (antes de one-hot)
                     # para mantener la proporción de clases si es desbalanceado.
)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del conjunto de validación: {X_val.shape[0]}")


# --- Definición del Modelo de Red Neuronal Densa ---
print("\nDefiniendo el modelo de red neuronal...")
# Usando la arquitectura que tenías, puedes experimentar con ella
model = models.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(NUM_ACTIONS, activation='softmax')
])
model.summary()
# --- Compilación del Modelo ---
print("\nCompilando el modelo...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Entrenamiento del Modelo ---
# ModelCheckpoint
CHECKPOINT_PATH = 'checkpoints/pong_bc_cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
if not os.path.exists(checkpoint_dir) and checkpoint_dir: # Asegurarse que checkpoint_dir no sea vacío
    os.makedirs(checkpoint_dir)
model_checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH, # Guardará el modelo completo (no solo pesos)
    save_weights_only=False, # Guardar el modelo completo
    monitor='val_accuracy',    # Métrica a monitorear
    mode='max',                # Guardar cuando la métrica monitoreada sea máxima
    save_best_only=True,       # Solo guardar el mejor modelo
    verbose=1
)

# ReduceLROnPlateau
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss', # Monitorear la pérdida de validación
    factor=0.2,         # Factor por el cual reducir LR: new_lr = lr * factor
    patience=5,         # Número de épocas sin mejora después de las cuales se reduce LR
    min_lr=0.0001,      # Límite inferior para la tasa de aprendizaje
    verbose=1
)

callbacks_list = [model_checkpoint_callback, reduce_lr_callback]

print(f"\nEntrenando el modelo durante {EPOCHS} épocas con batch_size {BATCH_SIZE}...")
try:
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list,
                        verbose=1)
except KeyboardInterrupt:
    pass

# --- Evaluación y Guardado del Modelo ---
print("\nEntrenamiento completado.")

loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Precisión en el conjunto de validación: {accuracy:.4f}")

print(f"\nGuardando el modelo entrenado en: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("Modelo guardado exitosamente.")
print("\nProceso de entrenamiento de Behavioral Cloning finalizado.")