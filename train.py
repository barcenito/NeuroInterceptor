import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import time
from functools import partial
import pickle

# Importamos tu entorno optimizado
from env_core import CombatDroneEnv, EnvParams, EnvState

# --- CONFIGURACIÓN DEL ENTRENAMIENTO ---
POP_SIZE = 2048         # Agentes (Debe ser divisible por TOP_K)
NUM_GENERATIONS = 1000   # Subido un poco para que aprenda a apuntar bien
TOP_K = 128             # Elites
MUTATION_POWER = 0.03   # Ruido
HIDDEN_SIZE = 64        # Subido a 64 para darle más capacidad cerebral

# --- 1. DEFINICIÓN DEL CEREBRO (RED NEURONAL) ---
class DronePilot(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input: 7 valores (Sensores)
        x = nn.Dense(HIDDEN_SIZE)(x)
        x = nn.tanh(x) 
        x = nn.Dense(HIDDEN_SIZE)(x)
        x = nn.tanh(x)
        
        # Output: 5 VALORES (AccelX, AccelY, AimX, AimY, Trigger)
        x = nn.Dense(5)(x)
        
        # Salida siempre entre -1 y 1
        return nn.tanh(x) 

# --- HELPER: CALCULAR OBSERVACIÓN ---
# Extraemos esto para usarlo al inicio y durante el bucle
def get_obs(state, params):
    safe_time = jnp.minimum(state.time_idx, 500 - 2)
    # Nota: Usamos params.trajectories directamente
    target_pos_now = params.trajectories[state.trajectory_idx, safe_time]
    target_pos_next = params.trajectories[state.trajectory_idx, safe_time + 1]
    target_vel = (target_pos_next - target_pos_now) * 60.0 # DT inversa
    
    rel_pos = target_pos_now - state.drone_pos
    rel_vel = target_vel - state.drone_vel
    
    obs = jnp.array([
        rel_pos[0], rel_pos[1],
        rel_vel[0], rel_vel[1],
        state.drone_vel[0], state.drone_vel[1],
        state.cooldown / 20.0
    ])
    return obs

# --- 2. ROLLOUT (SIMULACIÓN) ---
def rollout(rng, policy_params, env_params, env_static):
    # 1. Reset del entorno
    state = env_static.reset(rng, env_params)
    
    # 2. Calcular la PRIMERA observación manualmente
    initial_obs = get_obs(state, env_params)
    
    model = DronePilot()
    
    # Estructura de memoria para el bucle (Carry)
    # Llevamos: (Estado actual, Última observación)
    initial_carry = (state, initial_obs)

    def policy_step(carry, _):
        current_state, current_obs = carry
        
        # A. Inferencia (Cerebro)
        action = model.apply(policy_params, current_obs)
        
        # B. Paso del entorno (Física)
        # env.step ya nos devuelve la NUEVA observación
        next_state, next_obs, reward, done = env_static.step(current_state, action, env_params)
        
        # Si el episodio terminó, reward es 0 (masking)
        reward = jnp.where(current_state.done, 0.0, reward)
        
        new_carry = (next_state, next_obs)
        return new_carry, reward

    # Ejecutar 500 pasos
    final_carry, rewards = jax.lax.scan(policy_step, initial_carry, None, length=500)
    
    return jnp.sum(rewards)

# --- 3. CORE DE EVOLUCIÓN ---
def train():
    print(f"🚀 Iniciando entrenamiento EvoJAX (5 Outputs Aim/Move)...")
    
    # Setup
    env = CombatDroneEnv()
    env_params = env.default_params
    model = DronePilot()
    
    # Inicializar pesos
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((7,)) # 7 Sensores
    base_params = model.init(init_rng, dummy_obs)
    
    print(f"🧠 Red Neuronal inicializada. Capas: {list(base_params['params'].keys())}")

    # Función para crear población (Noise Injection)
    def create_pop_member(key, params):
        leaves, treedef = jax.tree_util.tree_flatten(params)
        # Ruido inicial fuerte (1.0) para diversidad
        noise = [jax.random.normal(key, l.shape) * 1.0 for l in leaves] 
        new_leaves = [l + n for l, n in zip(leaves, noise)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    # Crear claves y generar población inicial
    pop_keys = jax.random.split(rng, POP_SIZE)
    # vmap sobre keys (0), pero params se mantiene igual (None)
    population = jax.vmap(create_pop_member, in_axes=(0, None))(pop_keys, base_params)

    # JIT Rollout
    # vmap sobre: keys(0), population(0), env_params(None)
    batch_rollout = jax.jit(jax.vmap(partial(rollout, env_static=env), in_axes=(0, 0, None)))

    # Variables de seguimiento
    best_overall_score = -9999
    best_params_overall = None
    
    start_time = time.time()
    
    # --- BUCLE DE GENERACIONES ---
    for gen in range(NUM_GENERATIONS):
        gen_rng = jax.random.fold_in(rng, gen)
        
        # 1. Evaluar
        episode_keys = jax.random.split(gen_rng, POP_SIZE)
        fitness_scores = batch_rollout(episode_keys, population, env_params)
        
        # 2. Seleccionar (Top K)
        top_indices = jnp.argsort(fitness_scores)[::-1][:TOP_K]
        best_gen_score = fitness_scores[top_indices[0]]
        avg_score = jnp.mean(fitness_scores)
        
        if best_gen_score > best_overall_score:
            best_overall_score = best_gen_score
            # Guardamos copia exacta de los pesos del mejor
            best_params_overall = jax.tree_util.tree_map(lambda x: x[top_indices[0]], population)
        
        if gen % 10 == 0:
            print(f"Gen {gen:3d} | Best: {best_gen_score:6.2f} | Avg: {avg_score:6.2f} | 🏆 Global: {best_overall_score:6.2f}")
        
        # 3. Mutar y Repoblar
        elites = jax.tree_util.tree_map(lambda x: x[top_indices], population)
        
        # Cuántas copias de cada elite necesitamos
        n_copies = POP_SIZE // TOP_K 
        
        def mutate(params, key):
            leaves, treedef = jax.tree_util.tree_flatten(params)
            noise = [jax.random.normal(key, l.shape) * MUTATION_POWER for l in leaves]
            new_leaves = [l + n for l, n in zip(leaves, noise)]
            return jax.tree_util.tree_unflatten(treedef, new_leaves)

        # Generar claves para mutación: Shape (TOP_K, n_copies)
        mut_keys = jax.random.split(gen_rng, (TOP_K, n_copies))
        
        # Doble vmap: Primero por Elites, luego por Copias
        # Elite[i] se combina con mut_keys[i] (que tiene n_copies claves)
        next_gen_nested = jax.vmap(
            lambda elite, keys: jax.vmap(mutate, in_axes=(None, 0))(elite, keys)
        )(elites, mut_keys)
        
        # Aplanar la estructura anidada para volver a tener (POP_SIZE, ...)
        # next_gen_nested es (TOP_K, n_copies, LEAVES...)
        population = jax.tree_util.tree_map(
            lambda x: x.reshape((POP_SIZE,) + x.shape[2:]), 
            next_gen_nested
        )

    # --- FIN ---
    total_time = time.time() - start_time
    print(f"\n🏁 Entrenamiento finalizado en {total_time:.2f}s")
    
    # Guardar
    with open("best_brain.pkl", "wb") as f:
        pickle.dump(best_params_overall, f)
    print("💾 Cerebro guardado en 'best_brain.pkl'")

if __name__ == "__main__":
    train()