import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
from functools import partial 

# --- CONSTANTES ---
SCREEN_SIZE = 1.0
DT = 1.0 / 60.0
MAX_STEPS = 500
DRONE_DRAG = 0.95
THRUST_POWER = 2.5
BULLET_SPEED = 4.0      # Unidades por segundo
MAX_RANGE = 0.6         # Rango máximo de la bala
RELOAD_TIME = 20
AIM_CONE = 0.96         # Precisión requerida (aprox 16 grados)

@struct.dataclass
class EnvState:
	drone_pos: jnp.ndarray
	drone_vel: jnp.ndarray
	cooldown: jnp.int32
	trajectory_idx: jnp.int32
	time_idx: jnp.int32
	score: jnp.float32
	done: jnp.bool_
	steps: jnp.int32

@struct.dataclass
class EnvParams:
	trajectories: jnp.ndarray

class CombatDroneEnv:
	def __init__(self, dataset_path="targets_dataset.npy"):
		print(f"⚡ Inicializando entorno...")
		try:
			raw_data = jnp.load(dataset_path)
			self.trajectories = jnp.array(raw_data)
			print(f"✅ Dataset cargado: {self.trajectories.shape}")
		except:
			print("⚠️ Usando backup procedural.")
			# Generación simple de backup
			backup_data = []
			t = jnp.linspace(0, 10, 500)
			for i in range(10):
				x = 0.5 + 0.3 * jnp.sin(t + i)
				y = 0.5 + 0.3 * jnp.cos(t + i)
				backup_data.append(jnp.stack([x, y], axis=1))
			self.trajectories = jnp.array(backup_data)

	@property
	def default_params(self) -> EnvParams:
		return EnvParams(trajectories=self.trajectories)

	@property
	def action_shape(self):
		# [AccelX, AccelY, AimX, AimY, Trigger]
		return (5,)
	
	@property
	def obs_shape(self):
		return (7,)

	@partial(jax.jit, static_argnums=(0,))
	def reset(self, key: jax.random.PRNGKey, params: EnvParams) -> EnvState:
		k1, k2 = jax.random.split(key)
		traj_idx = jax.random.randint(k1, shape=(), minval=0, maxval=params.trajectories.shape[0])
		start_pos = jnp.array([0.5, 0.5]) + jax.random.uniform(k2, (2,), minval=-0.1, maxval=0.1)
		
		return EnvState(
			drone_pos=start_pos,
			drone_vel=jnp.zeros(2),
			cooldown=jnp.int32(0),
			trajectory_idx=traj_idx,
			time_idx=jnp.int32(0),
			score=jnp.float32(0.0),
			done=jnp.bool_(False),
			steps=jnp.int32(0)
		)

	@partial(jax.jit, static_argnums=(0,))
	def step(self, state: EnvState, action: jnp.ndarray, params: EnvParams) -> Tuple[EnvState, jnp.ndarray, jnp.float32, jnp.bool_]:
		
		# 1. POSICIÓN ACTUAL DEL OBJETIVO (Para sensores)
		safe_time = jnp.minimum(state.time_idx, MAX_STEPS - 2)
		target_pos_now = params.trajectories[state.trajectory_idx, safe_time]
		
		# 2. INTERPRETAR ACCIONES
		accel_cmd = action[:2] 
		aim_vec_raw = action[2:4] 
		trigger = action[4] > 0.0

		aim_norm = jnp.linalg.norm(aim_vec_raw) + 1e-6
		aim_dir = aim_vec_raw / aim_norm # Vector unitario de hacia dónde mira el cañón
		
		# 3. FÍSICA DEL DRON
		thrust = jnp.clip(accel_cmd, -1.0, 1.0) * THRUST_POWER
		new_vel = (state.drone_vel + thrust * DT) * DRONE_DRAG
		new_pos = state.drone_pos + new_vel * DT
		
		# Límites
		out_of_bounds = (new_pos < 0.0) | (new_pos > 1.0)
		new_pos = jnp.clip(new_pos, 0.0, 1.0)
		new_vel = jnp.where(out_of_bounds, 0.0, new_vel)

		# ========================================================
		# 4. CÁLCULO BALÍSTICO (MAGIA MATEMÁTICA)
		# ========================================================
		can_fire = state.cooldown <= 0
		
		# A. ¿A qué distancia está ahora?
		vec_now = target_pos_now - new_pos
		dist_now = jnp.linalg.norm(vec_now)
		
		# B. ¿Cuánto tardará la bala en llegar ahí?
		# t = distancia / velocidad
		flight_time = dist_now / BULLET_SPEED
		flight_frames = (flight_time / DT).astype(jnp.int32)
		
		# C. CONSULTAR ORÁCULO: ¿Dónde estará el objetivo en 'flight_frames'?
		future_idx = jnp.minimum(state.time_idx + flight_frames, MAX_STEPS - 1)
		target_pos_future = params.trajectories[state.trajectory_idx, future_idx]
		
		# D. VALIDAR IMPACTO FUTURO
		# El vector de tiro debe apuntar desde el Dron (ahora) hasta el Objetivo (futuro)
		vec_intercept = target_pos_future - new_pos
		dist_intercept = jnp.linalg.norm(vec_intercept) # Distancia real que recorre la bala
		dir_intercept = vec_intercept / (dist_intercept + 1e-6)
		
		# E. ¿Estamos alineados con ese punto futuro?
		# aim_dir: Hacia dónde mira el dron realmente
		# dir_intercept: Hacia dónde DEBERÍA mirar para acertar
		alignment_future = jnp.dot(aim_dir, dir_intercept)
		
		# CONDICIONES DE ACIERTO:
		# 1. Distancia Futura < MaxRange (Si el objetivo huye y sale de rango, fallas)
		# 2. Alineación Futura > Cono (Si apuntaste al futuro correctamente)
		in_range = dist_intercept < MAX_RANGE 
		aim_good = alignment_future > AIM_CONE 
		
		did_fire = trigger & can_fire
		is_hit = in_range & aim_good
		
		# --- RECOMPENSAS ---
		reward = 0.0
		
		# Recompensa por acercarse (usamos distancia actual para navegación)
		reward -= jnp.maximum(0.0, dist_now - MAX_RANGE) * 0.5
		
		# Recompensa de "Radar Lock" (Premio pequeño si alineas la predicción aunque no dispares)
		# Esto ayuda a la red a "entender" la predicción antes de arriesgarse a disparar
		reward += jnp.maximum(0.0, alignment_future) * 0.05
		
		# Resultado del Disparo
		if_hit_reward = 3.0   # Gran premio por predicción exitosa
		if_miss_reward = -2.5 # Castigo fuerte por desperdiciar munición
		
		shot_outcome = jnp.where(is_hit, if_hit_reward, if_miss_reward)
		reward += jnp.where(did_fire, shot_outcome, 0.0)
		
		# Penalizaciones menores
		reward -= 0.005 # Coste de tiempo
		is_camping = (new_pos < 0.05) | (new_pos > 0.95)
		reward -= jnp.sum(jnp.where(is_camping, 0.2, 0.0))
		
		# Penalización Anti-Spam (Si aprieta el gatillo sin balas)
		reward -= jnp.where(trigger & (state.cooldown > 0), 0.1, 0.0)

		# 5. ACTUALIZAR ESTADO
		new_cooldown = jnp.where(did_fire, RELOAD_TIME, state.cooldown - 1)
		new_steps = state.steps + 1
		done = new_steps >= MAX_STEPS
		
		new_state = state.replace(
			drone_pos=new_pos,
			drone_vel=new_vel,
			cooldown=new_cooldown,
			time_idx=new_steps,
			score=state.score + reward,
			done=done,
			steps=new_steps
		)
		
		# OBSERVACIONES (Sensores)
		# La red necesita saber VELOCIDAD RELATIVA para calcular la predicción
		safe_time_next = jnp.minimum(state.time_idx + 1, MAX_STEPS - 1)
		target_pos_next = params.trajectories[state.trajectory_idx, safe_time_next]
		
		# Velocidad del objetivo (Crítico para calcular 'Leading')
		target_vel = (target_pos_next - target_pos_now) / DT
		
		rel_pos = target_pos_now - new_pos
		rel_vel = target_vel - new_vel
		
		obs = jnp.array([
			rel_pos[0], rel_pos[1],     # Posición Relativa
			rel_vel[0], rel_vel[1],     # Velocidad Relativa (¡DATO CLAVE!)
			new_vel[0], new_vel[1],     # Mi Velocidad
			new_cooldown / RELOAD_TIME
		])
		
		return new_state, obs, reward, done

# --- CÓDIGO DE PRUEBA ---
if __name__ == "__main__":
	env = CombatDroneEnv()
	params = env.default_params
	rng = jax.random.PRNGKey(0)
	state = env.reset(rng, params)
	
	# ⚠️ IMPORTANTE: Ahora necesitamos 5 acciones
	# Accel(1,0), Aim(0,1), Trigger(1)
	action = jnp.array([1.0, 0.0,  0.0, 1.0,  1.0]) 
	
	print("Compilando Step...")
	start_compile = jax.numpy.arange(1) # Trigger JAX
	step_fn = env.step
	state, obs, reward, done = step_fn(state, action, params)
	
	print("✅ Step completado.")
	print(f"Reward del paso: {reward}")
	print(f"Obs Shape: {obs.shape}")