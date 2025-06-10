import pygame
from ple.games.pong import Pong
from ple import PLE
import time
import numpy as np
import os

# Asumimos que QAgentPong.py está en el mismo directorio o en PYTHONPATH
from QAgentPong import QAgent # Asegúrate de que esto funcione

# --- Configuración ---
FPS = 30
SCREEN_WIDTH = 200  # Debe coincidir con el juego con el que se entrenó el Q-Agent
SCREEN_HEIGHT = 320 # Debe coincidir con el juego con el que se entrenó el Q-Agent
MAX_SCORE_GAME = 5
MAX_STEPS_PER_EPISODE = 5000
SCORE_CRITERIA_TO_SAVE = 10 # Puntaje para guardar ejemplos del episodio
Q_TABLE_PATH = "pong_q_table_final.pkl" # Ruta a tu Q-table entrenada
DATASET_FILENAME = 'pong_q_expert_dataset.npy'
NUM_EPISODES_COLLECT = 1000 # Cuántos episodios jugar para recolectar datos

# Acciones que el Q-agent puede tomar y cómo las mapearemos para el dataset de BC
ACTIONS_MAP_TO_IDX = {
    119: 0,    # Subir
    115: 1,    # Bajar
    None: 2    # Quedarse quieto
}

def collect_data_from_q_expert(num_episodes=NUM_EPISODES_COLLECT):
    """
    Usa un Q-agent entrenado para jugar Pong y guarda los pares (estado_normalizado, accion_idx)
    SOLO de los episodios donde el agente alcanza el SCORE_CRITERIA_TO_SAVE.
    """
    # Inicializar el juego
    game = Pong(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, MAX_SCORE=MAX_SCORE_GAME)
    env = PLE(game, display_screen=False, fps=30)
    env.init()

    ple_actions = env.getActionSet()

    if not os.path.exists(Q_TABLE_PATH):
        print(f"Error: No se encontró la Q-table en {Q_TABLE_PATH}")
        return
    
    expert_agent = QAgent(game, ple_actions, load_q_table_path=Q_TABLE_PATH)
    expert_agent.epsilon = 0.0

    print(f"Q-Agent cargado desde: {Q_TABLE_PATH}")
    print(f"Recolectando datos durante un máximo de {num_episodes} episodios.")
    print(f"Se guardarán solo episodios donde el score del experto sea: {SCORE_CRITERIA_TO_SAVE}")


    all_collected_data = [] # Lista final para todos los datos de episodios válidos
    episodes_processed = 0
    episodes_saved_count = 0

    for episode_idx in range(num_episodes):
        episodes_processed += 1
        env.reset_game()
        current_ple_state_dict = env.getGameState()
        done = False
        
        current_episode_data = [] # Datos temporales para este episodio
        steps_this_episode = 0
        total_reward_episode = 0

        while not done:
            # Elegir acción del experto (Q-Agent)
            expert_action_ple = expert_agent.choose_action(current_ple_state_dict)
            
            # Normalizar el estado actual y crear el ejemplo
            state_features = np.array([
                current_ple_state_dict['player_y'] / SCREEN_HEIGHT,
                current_ple_state_dict['ball_x'] / SCREEN_WIDTH,
                current_ple_state_dict['ball_y'] / SCREEN_HEIGHT,
                current_ple_state_dict['ball_velocity_x'] / SCREEN_WIDTH,
                current_ple_state_dict['ball_velocity_y'] / SCREEN_HEIGHT
            ])
            action_idx = ACTIONS_MAP_TO_IDX[expert_action_ple]
            current_episode_data.append((state_features, action_idx))

            total_reward_episode += env.act(expert_action_ple)
            current_ple_state_dict = env.getGameState()
            done = env.game_over()
            
            steps_this_episode += 1
            if steps_this_episode >= MAX_STEPS_PER_EPISODE:
                print(f"Episodio {episode_idx + 1}: Máximo de pasos alcanzado ({MAX_STEPS_PER_EPISODE}).")
                break

            # Manejo de eventos de Pygame (necesario incluso sin display_screen para evitar cuelgues en algunos sistemas)
            # y para permitir interrupción limpia
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("Recolección interrumpida por el usuario.")
                    if all_collected_data:
                        np.save(DATASET_FILENAME, np.array(all_collected_data, dtype=object))
                        print(f"Datos parciales ({len(all_collected_data)} muestras de {episodes_saved_count} episodios) guardados en {DATASET_FILENAME}")
                    return
        
        if total_reward_episode >= SCORE_CRITERIA_TO_SAVE:
            all_collected_data.extend(current_episode_data)
            episodes_saved_count += 1
            print(f"Episodio {episode_idx + 1}: Score {total_reward_episode} - GUARDADO. Total muestras acumuladas: {len(all_collected_data)}")
        else:
            # print(f"Episodio {episode_idx + 1}: Score {final_player_score} - DESCARTADO (se requiere {SCORE_CRITERIA_TO_SAVE}).") # Verboso
            pass
        
        if (episode_idx + 1) % 50 == 0: # Imprimir progreso cada 50 episodios
             print(f"Progreso: {episode_idx + 1}/{num_episodes} episodios procesados. {episodes_saved_count} episodios guardados.")


    print(f"\nRecolección de datos terminada.")
    print(f"Episodios procesados: {episodes_processed}")
    print(f"Episodios que cumplieron el criterio (score={SCORE_CRITERIA_TO_SAVE}): {episodes_saved_count}")
    
    if all_collected_data:
        dataset_np = np.array(all_collected_data, dtype=object)
        np.save(DATASET_FILENAME, dataset_np)
        print(f"Total de {len(dataset_np)} muestras guardadas en {DATASET_FILENAME}")
    else:
        print(f"No se guardaron datos. Ningún episodio alcanzó el score de {SCORE_CRITERIA_TO_SAVE}.")
    
    pygame.quit()

if __name__ == '__main__':
    print("--- Recolección de Datos del Experto (Q-Agent) para Behavioral Cloning ---")
    
    # Verificar si el archivo Q-table existe antes de empezar
    if not os.path.exists(Q_TABLE_PATH):
        print(f"Error crítico: La Q-table '{Q_TABLE_PATH}' no existe.")
        print("Por favor, asegúrate de que la ruta es correcta y el archivo está presente.")
    else:
        collect_data_from_q_expert(num_episodes=NUM_EPISODES_COLLECT)

        # Pequeña verificación de lo guardado (opcional)
        if os.path.exists(DATASET_FILENAME):
            print(f"\nVerificando el archivo guardado: {DATASET_FILENAME}")
            loaded_data = np.load(DATASET_FILENAME, allow_pickle=True)
            print(f"Forma del dataset cargado: {loaded_data.shape}")
            if loaded_data.shape[0] > 0:
                print("Ejemplo de la primera muestra:")
                print(f"  Estado (features): {loaded_data[0][0]}")
                print(f"  Acción (índice): {loaded_data[0][1]}")
                print(f"Tipo de dato de las features: {type(loaded_data[0][0])}")
                print(f"Tipo de dato del índice de acción: {type(loaded_data[0][1])}")
            else:
                print("El dataset guardado está vacío.")