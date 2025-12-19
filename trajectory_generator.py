import numpy as np
import pygame
import sys

# --- CONFIGURACIÓN ---
NUM_TRAJECTORIES = 1000   # Cuántas rutas diferentes vamos a pre-calcular
DURATION_FRAMES = 500     # Duración de cada "partida"
DT = 1.0 / 60.0           # 60 FPS
CANVAS_SIZE = 1.0         # El mapa va de 0.0 a 1.0

# Archivo de salida
OUTPUT_FILE = "targets_dataset.npy"

# Configuración de Pygame
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
FPS = 60
TRAIL_LENGTH = 100

# Colores
COLOR_BG = (20, 20, 30)
COLOR_GRID = (50, 50, 70)
COLOR_TARGET = (255, 50, 50)
COLOR_TRAIL = (255, 100, 100)
COLOR_TEXT = (200, 200, 200)

def generate_smooth_path(seed, num_frames):
    """
    Genera una trayectoria suave usando superposición de ondas senoidales (Lissajous aleatorio).
    Esto simula un vuelo orgánico (como una mosca o un dron).
    """
    np.random.seed(seed)
    t = np.linspace(0, num_frames * DT, num_frames)
    
    # Parámetros aleatorios para Eje X
    freq_x1, amp_x1, phase_x1 = np.random.uniform(0.2, 1.5), np.random.uniform(0.1, 0.4), np.random.uniform(0, 2*np.pi)
    freq_x2, amp_x2, phase_x2 = np.random.uniform(1.5, 3.0), np.random.uniform(0.05, 0.1), np.random.uniform(0, 2*np.pi)
    
    # Parámetros aleatorios para Eje Y
    freq_y1, amp_y1, phase_y1 = np.random.uniform(0.2, 1.5), np.random.uniform(0.1, 0.4), np.random.uniform(0, 2*np.pi)
    freq_y2, amp_y2, phase_y2 = np.random.uniform(1.5, 3.0), np.random.uniform(0.05, 0.1), np.random.uniform(0, 2*np.pi)
    
    # Generar posición X (Suma de dos ondas para complejidad)
    x = 0.5 + amp_x1 * np.sin(freq_x1 * t + phase_x1) + amp_x2 * np.sin(freq_x2 * t + phase_x2)
    
    # Generar posición Y
    y = 0.5 + amp_y1 * np.cos(freq_y1 * t + phase_y1) + amp_y2 * np.cos(freq_y2 * t + phase_y2)
    
    # Clampear para que no se salga del mapa (dejando un margen del 5%)
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    
    # Retornar forma (Frames, 2) -> [[x0, y0], [x1, y1]...]
    return np.stack([x, y], axis=1)

def create_dataset():
    print(f"Generando {NUM_TRAJECTORIES} trayectorias de {DURATION_FRAMES} frames...")
    dataset = []
    
    for i in range(NUM_TRAJECTORIES):
        # Usamos i como semilla para que sea reproducible
        path = generate_smooth_path(seed=i, num_frames=DURATION_FRAMES)
        dataset.append(path)
    
    dataset = np.array(dataset, dtype=np.float32)
    
    # Guardar en disco
    np.save(OUTPUT_FILE, dataset)
    print(f"✅ Dataset guardado en '{OUTPUT_FILE}'. Shape: {dataset.shape}")
    print(f"   (Batch, Time, Coords) = {dataset.shape}")
    return dataset

def normalize_to_screen(pos, screen_width, screen_height):
    """Convierte coordenadas normalizadas [0,1] a píxeles de pantalla."""
    x, y = pos
    screen_x = int(x * screen_width)
    screen_y = int((1.0 - y) * screen_height)  # Invertir Y para que sea estándar en gráficos
    return (screen_x, screen_y)

def visualize_trajectory(trajectory_data, trajectory_idx=0):
    """
    Visualizador mejorado con Pygame.
    
    Controles:
    - ESPACIO: Pausar/Reanudar
    - FLECHA DERECHA: Siguiente frame
    - FLECHA IZQUIERDA: Frame anterior
    - R: Reiniciar
    - Q o ESC: Salir
    """
    pygame.init()
    
    # Crear ventana
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Validación de Trayectoria - Trayectoria #{trajectory_idx}")
    clock = pygame.time.Clock()
    font_small = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 32)
    
    # Datos
    path = trajectory_data  # Shape (Frames, 2)
    current_frame = 0
    paused = False
    running = True
    
    print(f"🎥 Visualizador Pygame iniciado. Trayectoria #{trajectory_idx}")
    print("   Controles: ESPACIO=Pausa, FLECHAS=Navegar, R=Reiniciar, Q=Salir")
    
    while running:
        dt = clock.tick(FPS) / 1000.0  # Delta time en segundos
        
        # Manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    current_frame = min(current_frame + 1, len(path) - 1)
                elif event.key == pygame.K_LEFT:
                    current_frame = max(current_frame - 1, 0)
                elif event.key == pygame.K_r:
                    current_frame = 0
                    paused = False
        
        # Avanzar frame si no está pausado
        if not paused:
            current_frame += 1
            if current_frame >= len(path):
                current_frame = 0  # Reiniciar al final
        
        # Dibujar
        screen.fill(COLOR_BG)
        
        # Dibujar grid
        grid_spacing = WINDOW_WIDTH // 5
        for i in range(6):
            # Líneas verticales
            pygame.draw.line(screen, COLOR_GRID, (i * grid_spacing, 0), (i * grid_spacing, WINDOW_HEIGHT), 1)
            # Líneas horizontales
            pygame.draw.line(screen, COLOR_GRID, (0, i * grid_spacing), (WINDOW_WIDTH, i * grid_spacing), 1)
        
        # Dibujar rastro
        trail_start = max(0, current_frame - TRAIL_LENGTH)
        trail_points = []
        for idx in range(trail_start, current_frame + 1):
            pos_screen = normalize_to_screen(path[idx], WINDOW_WIDTH, WINDOW_HEIGHT)
            trail_points.append(pos_screen)
        
        if len(trail_points) > 1:
            pygame.draw.lines(screen, COLOR_TRAIL, False, trail_points, 2)
        
        # Dibujar objetivo actual
        current_pos = normalize_to_screen(path[current_frame], WINDOW_WIDTH, WINDOW_HEIGHT)
        pygame.draw.circle(screen, COLOR_TARGET, current_pos, 8)
        pygame.draw.circle(screen, COLOR_TRAIL, current_pos, 12, 2)
        
        # Dibujar información
        status_text = "PAUSA" if paused else "REPRODUCIENDO"
        status_surface = font_small.render(status_text, True, COLOR_TEXT)
        screen.blit(status_surface, (10, 10))
        
        frame_text = font_small.render(f"Frame: {current_frame + 1}/{len(path)}", True, COLOR_TEXT)
        screen.blit(frame_text, (10, 40))
        
        time_text = font_small.render(f"Tiempo: {current_frame * DT:.2f}s", True, COLOR_TEXT)
        screen.blit(time_text, (10, 70))
        
        # Mostrar posición actual
        x, y = path[current_frame]
        pos_text = font_small.render(f"Pos: ({x:.3f}, {y:.3f})", True, COLOR_TEXT)
        screen.blit(pos_text, (10, 100))
        
        # Mostrar controles
        controls = [
            "ESPACIO: Pausa",
            "Flechas: Navegar",
            "R: Reiniciar",
            "Q: Salir"
        ]
        for i, control in enumerate(controls):
            ctrl_surface = font_small.render(control, True, COLOR_TEXT)
            screen.blit(ctrl_surface, (WINDOW_WIDTH - 220, 10 + i * 30))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    # 1. Generar y guardar
    data = create_dataset()
    
    # 2. Visualizar la primera trayectoria para comprobar (Índice 0)
    # Cambia el índice por otro número para ver diferentes trayectorias
    visualize_trajectory(data[0], trajectory_idx=0)