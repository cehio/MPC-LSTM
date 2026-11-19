# MPC-LSTM
Realiza un dataset en base a un controlador MPC para que una red LSTM imite el comportamiento del controlador 

# Simulador de Control Predictivo

Este proyecto implementa un simulador de alta fidelidad para el control de temperatura y humedad en un prototipo de invernadero utilizando **Control Predictivo Basado en Modelo** con linealización sucesiva. 

El sistema está diseñado para generar datos de simulación robustos, incluyendo ruido de sensor, retardos (lag) y degradación de actuadores, lo que lo hace ideal para validar estrategias de control o generar datasets para entrenamiento de modelos de Machine Learning (Data Augmentation).

## Características Principales

- **Modelo No Lineal de la Planta:** Dinámica de temperatura y humedad psicrométrica resuelta con método de Runge-Kutta de 4º orden (RK4).
- **MPC con Linealización Sucesiva:** Jacobiano numérico en tiempo continuo, discretización exacta (ZOH) y formulación como Problema Cuadrático (QP) usando `CVXPY`.
- **Gestión de Restricciones:** Límites duros en actuadores (calefactor, humidificador, ventilación) y restricciones de tasa de cambio (Slew Rate) para evitar desgaste y *chattering*.
- **Simulación de Sensores Reales:** Incluye modelo de retardo de primer orden (lag) y ruido gaussiano para emular sensores tipo AHT10.
- **Biblioteca de Escenarios:** Más de 40 escenarios predefinidos que cubren estaciones (Invierno, Verano, Primavera, Otoño), transitorios y casos límite de humedad externa.
- **Data Augmentation:** Capacidad de inyectar perturbaciones multiplicativas en los actuadores (ej. degradación al 85% o sobrepotencia al 115%) para probar la robustez del controlador.
- **Modo Rápido:** Bandera `--fast` para ejecutar un subconjunto representativo de escenarios en ~30 minutos, ideal para desarrollo iterativo.

## 📋 Requisitos Previos

- Python 3.8 o superior.
- Se recomienda el uso de un entorno virtual.

## ⚙️ Instalación

1. Clona o descarga este repositorio.
2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
