import pygame
import sys
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# Importamos tu entorno
from env_core import CombatDroneEnv, EnvParams, EnvState

# --- CONFIGURACIÓN ESTÉTICA ---
WIDTH, HEIGHT = 900, 900
FPS = 60

# Configuración Física (DEBE SER IGUAL A ENV_CORE)
BULLET_SPEED_SIM = 4.0  # Unidades por segundo (Debe coincidir con env_core.py)
MAX_RANGE_SIM = 0.5

# Paleta Neon
C_BG = (10, 12, 20)         # Navy Dark
C_DRONE = (0, 255, 255)     # Cyan
C_TARGET = (255, 50, 80)    # Red Pink
C_GHOST = (255, 255, 255)   # White (Target Futuro)
C_BULLET = (100, 255, 100)  # Green Laser
C_AIM_LINE = (50, 50, 100)  # Dim Blue
C_LOCKED = (0, 255, 0)      # Bright Green

# --- RED NEURONAL ---
HIDDEN_SIZE = 64 

class DronePilot(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(HIDDEN_SIZE)(x)
        x = nn.tanh(x)
        x = nn.Dense(HIDDEN_SIZE)(x)
        x = nn.tanh(x)
        x = nn.Dense(5)(x) # [AccelX, AccelY, AimX, AimY, Trigger]
        return nn.tanh(x)

# --- CLASE BALA REALISTA CON VALIDACIÓN DE RANGO ---
class VisualBullet:
    def __init__(self, start_pos, aim_dir, target_pos_now, drone_vel, target_vel, max_range=0.4):
        """
        start_pos: Posición del dron (0-1)
        aim_dir: Vector normalizado hacia donde apunta
        target_pos_now: Posición actual del objetivo
        drone_vel: Velocidad del dron
        target_vel: Velocidad del objetivo
        max_range: Rango máximo de la bala
        """
        self.start_pos = np.array(start_pos, dtype=float)
        self.pos = np.array(start_pos, dtype=float)
        self.dir = np.array(aim_dir, dtype=float)
        self.max_range = max_range
        
        # CÁLCULO DE VELOCIDAD
        self.speed_per_frame = BULLET_SPEED_SIM / FPS
        self.distance_traveled = 0.0  # Rastreamos distancia recorrida
        
        # Guardamos info del objetivo para calcular colisión
        self.target_pos_now = np.array(target_pos_now, dtype=float)
        self.target_vel = np.array(target_vel, dtype=float)
        
        # Velocidad relativa (para predicción)
        self.rel_vel = self.target_vel - np.array(drone_vel, dtype=float)
        
        # Tiempo estimado de vuelo
        dist_now = np.linalg.norm(self.target_pos_now - self.start_pos)
        self.estimated_flight_time = dist_now / BULLET_SPEED_SIM if BULLET_SPEED_SIM > 0 else 0
        self.elapsed_time = 0.0
        
        self.alive = True
        self.hit_target = False

    def update(self, current_target_pos):
        """Actualiza posición y chequea colisión"""
        if not self.alive:
            return
        
        # Movimiento de la bala
        self.pos += self.dir * self.speed_per_frame
        self.distance_traveled += self.speed_per_frame
        self.elapsed_time += 1.0 / FPS
        
        # ===== VALIDACIÓN 1: ¿Superó el rango máximo? =====
        if self.distance_traveled > self.max_range:
            self.alive = False
            return
        
        # ===== VALIDACIÓN 2: ¿Salió de la pantalla? =====
        if not (0 <= self.pos[0] <= 1 and 0 <= self.pos[1] <= 1):
            self.alive = False
            return
        
        # ===== VALIDACIÓN 3: ¿Chocó con el objetivo? =====
        # Usamos un radio de colisión pequeño
        HIT_RADIUS = 0.03
        dist_to_target = np.linalg.norm(self.pos - current_target_pos)
        
        if dist_to_target < HIT_RADIUS:
            self.alive = False
            self.hit_target = True

    def draw(self, screen):
        if not self.alive:
            return
            
        px_pos = (int(self.pos[0] * WIDTH), int(self.pos[1] * HEIGHT))
        
        # Color según estado
        color = C_BULLET
        if self.hit_target:
            color = (255, 255, 100)  # Amarillo explosión
        
        # Proyectil
        pygame.draw.circle(screen, color, px_pos, 4)
        
        # Estela proporcional al rango consumido
        intensity = int(200 * (1.0 - self.distance_traveled / self.max_range))
        tail_len = 15
        px_tail = (
            int(px_pos[0] - self.dir[0] * tail_len),
            int(px_pos[1] - self.dir[1] * tail_len)
        )
        pygame.draw.line(screen, (100, intensity, 100), px_pos, px_tail, 2)
        
        # Barra de rango visual (arriba del proyectil)
        range_remaining = 1.0 - (self.distance_traveled / self.max_range)
        bar_width = int(30 * range_remaining)
        pygame.draw.line(screen, (0, 200, 0), 
                        (px_pos[0] - 15, px_pos[1] - 20),
                        (px_pos[0] - 15 + bar_width, px_pos[1] - 20), 2)

# --- UTILIDADES ---
def to_screen(pos):
    return int(pos[0] * WIDTH), int(pos[1] * HEIGHT)

def run_visualizer(model_path="best_brain.pkl"):
    # 1. Cargar Cerebro
    try:
        with open(model_path, "rb") as f:
            best_params = pickle.load(f)
        print("🧠 Cerebro cargado.")
    except Exception as e:
        print(f"❌ Error cargando {model_path}: {e}")
        return

    # 2. Inicializar
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEURO-INTERCEPTOR | Predictive Aiming Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    # Entorno JAX
    env = CombatDroneEnv()
    env_params = env.default_params
    model = DronePilot()

    @jax.jit
    def predict(params, obs):
        return model.apply(params, obs)

    rng = jax.random.PRNGKey(42)
    state = env.reset(rng, env_params)
    
    # Init dummy
    dummy_action = jnp.zeros(5)
    state, obs, _, _ = env.step(state, dummy_action, env_params)

    bullets = []
    trail = []
    total_score = 0
    running = True

    print("🚀 Visualizador: Círculo BLANCO = Dónde debes apuntar (Futuro)")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    rng = jax.random.split(rng)[0]
                    state = env.reset(rng, env_params)
                    trail = []
                    bullets = []
                    total_score = 0

        # --- 1. PROCESAMIENTO ---
        action = predict(best_params, obs)
        np_action = np.array(action)
        
        # Desempaquetar
        thrust = np_action[:2]
        aim_raw = np_action[2:4]
        trigger = np_action[4]
        
        # Vector de Mira Normalizado
        aim_norm = np.linalg.norm(aim_raw) + 1e-6
        aim_dir = aim_raw / aim_norm

        # --- 2. CÁLCULOS DE VISUALIZACIÓN PREDICTIVA (ORÁCULO) ---
        # Queremos saber dónde estará el objetivo cuando la bala llegue
        
        # A. Distancia actual
        safe_time = min(int(state.time_idx), 499)
        pos_drone_np = np.array(state.drone_pos)
        pos_target_now = np.array(env_params.trajectories[state.trajectory_idx, safe_time])
        
        # Calcular velocidad del objetivo
        safe_time_next = min(safe_time + 1, 499)
        pos_target_next = np.array(env_params.trajectories[state.trajectory_idx, safe_time_next])
        target_vel = (pos_target_next - pos_target_now) * FPS  # FPS para convertir a unidades/segundo
        
        dist_now = np.linalg.norm(pos_target_now - pos_drone_np)
        
        # B. Tiempo de vuelo
        time_to_hit = dist_now / BULLET_SPEED_SIM
        frames_ahead = int(time_to_hit * 60) # 60 ticks por segundo
        
        # C. Posición Futura (Ghost Target)
        future_time = min(safe_time + frames_ahead, 499)
        pos_target_future = np.array(env_params.trajectories[state.trajectory_idx, future_time])

        # --- 3. PASO DE SIMULACIÓN ---
        next_state, next_obs, reward, done = env.step(state, action, env_params)
        total_score += float(reward)

        # Disparo (Si trigger > 0 y cooldown == 0)
        # Nota: Usamos state.cooldown anterior al step para ser precisos
        if trigger > 0 and state.cooldown <= 0:
            # Pasamos la info de predicción a la bala
            bullets.append(VisualBullet(
                start_pos=state.drone_pos,
                aim_dir=aim_dir,
                target_pos_now=pos_target_now,
                drone_vel=state.drone_vel,
                target_vel=target_vel,
                max_range=0.4  # Coincide con MAX_RANGE en env_core.py
            ))

        # --- ACTUALIZAR Y RENDERIZAR BALAS ---
        for b in bullets:
            b.update(pos_target_now)  # Pasamos posición actual del objetivo
            b.draw(screen)

        bullets = [b for b in bullets if b.alive]

        # --- 4. RENDERIZADO ---
        screen.fill(C_BG)

        # Coordenadas Pantalla
        px_drone = to_screen(pos_drone_np)
        px_target = to_screen(pos_target_now)
        px_ghost = to_screen(pos_target_future)

        # A. DIBUJAR RASTRO
        trail.append(px_target)
        if len(trail) > 40: trail.pop(0)
        if len(trail) > 2:
            pygame.draw.lines(screen, (50, 0, 0), False, trail, 2)

        # B. DIBUJAR OBJETIVO ACTUAL (Rojo)
        pygame.draw.circle(screen, C_TARGET, px_target, 8)
        
        # C. DIBUJAR GHOST TARGET (Blanco Hueco) -> ¡AQUÍ ES DONDE DEBE DISPARAR!
        # Dibujamos una línea punteada desde el objetivo actual al futuro
        pygame.draw.line(screen, (100, 100, 100), px_target, px_ghost, 1)
        pygame.draw.circle(screen, C_GHOST, px_ghost, 8, 1) # Hueco

        # D. DIBUJAR DRON
        pygame.draw.circle(screen, C_DRONE, px_drone, 10)

        # E. VISUALIZAR MIRA Y PREDICCIÓN
        # Calculamos alineación con el FANTASMA (no con el actual)
        vec_to_ghost = pos_target_future - pos_drone_np
        vec_to_ghost /= (np.linalg.norm(vec_to_ghost) + 1e-6)
        
        # Producto punto con la mira
        alignment_ghost = np.dot(aim_dir, vec_to_ghost)
        
        # Línea de Mira (Infinita)
        aim_end_px = (
            int(px_drone[0] + aim_dir[0] * 1000), 
            int(px_drone[1] + aim_dir[1] * 1000)
        )
        
        # Color de la mira:
        # Rojo: Apuntando mal
        # Amarillo: Apuntando al objetivo actual (Mal, fallarás por retardo)
        # Verde: Apuntando al futuro (BIEN!)
        line_color = (100, 100, 100) # Gris por defecto
        
        # Alineación con objetivo actual (para debug)
        vec_to_now = pos_target_now - pos_drone_np
        vec_to_now /= (np.linalg.norm(vec_to_now) + 1e-6)
        align_now = np.dot(aim_dir, vec_to_now)

        if align_now > 0.96:
            line_color = (255, 255, 0) # Amarillo (Lag Shot warning)
        
        if alignment_ghost > 0.96:
            line_color = C_LOCKED # Verde (PERFECT PREDICTION)

        pygame.draw.line(screen, line_color, px_drone, aim_end_px, 1)
        
        # F. BALAS (Ya actualizadas y dibujadas arriba)
        for b in bullets:
            b.draw(screen)

        # UI
        ui = [
            f"PREDICTION DELTA: {frames_ahead} frames",
            f"LOCK FUTURE: {alignment_ghost:.3f}",
            f"SCORE: {total_score:.1f}",
            "AMARILLO = Apuntando al presente (FALLO)",
            "VERDE    = Apuntando al futuro (ACIERTO)"
        ]
        
        for i, t in enumerate(ui):
            col = C_GHOST
            if "VERDE" in t and alignment_ghost > 0.96: col = C_LOCKED
            surf = font.render(t, True, col)
            screen.blit(surf, (10, 10 + i*15))

        pygame.display.flip()
        clock.tick(FPS)
        
        state = next_state
        obs = next_obs
        
        if done:
            pygame.time.delay(100)
            rng = jax.random.split(rng)[0]
            state = env.reset(rng, env_params)
            total_score = 0
            trail = []
            bullets = []

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_visualizer()