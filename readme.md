# Neuro-Interceptor: EvoJAX Combat Drone

**Neuro-Interceptor** es un proyecto de investigación en Inteligencia Artificial que utiliza Computación Evolutiva (Neuroevolución) acelerada por hardware (JAX/EvoJAX) para entrenar una red neuronal capaz de pilotar un dron de combate en un entorno de simulación 2D.

##  Objetivo
Desarrollar un agente autónomo capaz de interceptar objetivos aéreos maniobrables mediante:
1.  **Persecución:** Navegación física con inercia para entrar en rango efectivo.
2.  **Predicción:** Cálculo implícito de la posición futura del objetivo.
3.  **Disparo:** Ejecución de disparo balístico compensando el retardo del proyectil.

##  Especificaciones del Entorno

### 1. El Agente (Dron)
*   **Física:** Modelo Newtoniano continuo (Inercia, Aceleración, Drag).
*   **Armamento:** Cañón con velocidad de proyectil finita y tiempo de recarga (Cooldown).
*   **Restricción de Rango ($D_{max}$):** El arma solo es efectiva si la distancia al objetivo es menor a un umbral definido. Disparar desde lejos resulta en fallo automático.
*   **Sensores (Inputs):** Telemetría relativa (Lidar/Radar simulado). El agente **no** conoce su posición absoluta (x,y), solo vectores relativos hacia el objetivo y su propia velocidad.

### 2. El Objetivo (Target)
*   **Comportamiento:** Trayectorias complejas pre-calculadas (Offline).
*   **Movimiento:** Curvas suaves, cambios de dirección y velocidad variable (simulando evasión).
*   **Generación:** Basada en combinaciones armónicas (Lissajous) y ruido procedimental para garantizar coherencia física.

##  Arquitectura del Sistema

### Stack Tecnológico
*   **Simulación & Entrenamiento:** Python + JAX (Aceleración GPU/TPU).
*   **Gestión de Datos:** Numpy (Generación de datasets offline).
*   **Visualización:** Matplotlib (Validación y Renderizado).

### Módulos
1.  **`trajectory_generator.py`:** Script offline que genera miles de rutas de vuelo válidas y las exporta a `.npy`.
2.  **`env_core.py` (JAX):** Motor de física optimizado. Carga el `.npy` en la VRAM y ejecuta miles de simulaciones en paralelo (`vmap`).
3.  **`visualizer.py`:** Herramienta para renderizar el historial de estados y validar el comportamiento de la IA y la física.

##  Criterios de Éxito (Fitness)
La evolución se guiará por una función de recompensa compuesta:
1.  **Recompensa de Aproximación:** Si $Distancia > Rango$, premiar la reducción de distancia.
2.  **Recompensa de Puntería/Impacto:** Premiar el impacto predictivo exitoso.
3.  **Eficiencia:** Penalización por disparos fallidos o colisiones con los límites del mapa.