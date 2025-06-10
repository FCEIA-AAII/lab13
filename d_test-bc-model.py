import pygame
from pygame.locals import K_w, K_s
import ple
from ple.games.pong import Pong
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

# --- Configuración del Entorno y Agente de Behavioral Cloning ---
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 320
MAX_SCORE_GAME = 5
MODEL_PATH = 'pong_bc_q_expert_model.h5' # Ruta a tu modelo entrenado
AGENT_ACTIONS_FROM_IDX = [K_w, K_s, None] # 0: Subir, 1: Bajar, 2: Quieto

# Cargar el modelo de Behavioral Cloning
if not os.path.exists(MODEL_PATH):
    print(f"Error: No se encontró el modelo en {MODEL_PATH}.")
    exit()

print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    bc_model = load_model(MODEL_PATH)
    print("Modelo de Behavioral Cloning cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Inicializar el juego Pong
game = Pong(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, MAX_SCORE=MAX_SCORE_GAME)
env = ple.PLE(game, display_screen=True, fps=30) # fps=30, display_screen=True para ver
env.init()

# --- Ejecución del Agente de Behavioral Cloning ---
print("\n--- Ejecutando agente de Behavioral Cloning ---")
num_episodes_to_play = 5 # Probar 5 episodios, como en c_test-pong.py

for episode in range(num_episodes_to_play):
    env.reset_game()
    current_ple_state_dict = env.getGameState()
    done = False
    total_reward_episode = 0

    print(f"Iniciando episodio de prueba {episode + 1}/{num_episodes_to_play}")

    while not done:
        # 1. Obtener el estado actual y convertirlo a features
        state_features = np.array([
            current_ple_state_dict['player_y'] / SCREEN_HEIGHT,
            current_ple_state_dict['ball_x'] / SCREEN_WIDTH,
            current_ple_state_dict['ball_y'] / SCREEN_HEIGHT,
            current_ple_state_dict['ball_velocity_x'] / SCREEN_WIDTH,
            current_ple_state_dict['ball_velocity_y'] / SCREEN_HEIGHT
        ]).reshape(1, -1) # Reshape para (1, num_features)

        # 2. Predecir la acción usando el modelo de BC
        action_probabilities = bc_model.predict(state_features, verbose=0)[0]
        predicted_action_idx = np.argmax(action_probabilities)
        
        # 3. Mapear el índice de acción a la acción de PLE
        ple_action = AGENT_ACTIONS_FROM_IDX[predicted_action_idx]

        # 4. Ejecutar la acción en el entorno
        reward = env.act(ple_action)
        
        # Actualizar estado y done
        current_ple_state_dict = env.getGameState()
        done = env.game_over()
        
        total_reward_episode += reward

        # Manejo de eventos de Pygame para poder cerrar la ventana
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print("Juego interrumpido por el usuario.")
                exit()
        
        time.sleep(0.03) # Pausa para visualización, igual que en c_test-pong.py

    print(f"Episodio {episode+1} terminado. Recompensa total: {total_reward_episode}")

pygame.quit()
print("\n--- Fin del juego del agente de Behavioral Cloning ---")