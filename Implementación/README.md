# Inferencia y Control Ambiental en Tiempo Real con LSTM (ESP32)

Script en Python para ejecutar el un modelo **LSTM** entrenado (imitación de MPC) y enviar comandos a un **ESP32** vía **puerto serial**. Mientras acumula las primeras `10` lecturas históricas (`SEQUENCE_LENGTH = 10`), aplica un control de respaldo simple (`apply_simple_control`) y, una vez completa la ventana temporal, usa las predicciones del modelo LSTM para accionar ventana, calentador y humidificador.

---

## Componentes del sistema

* **`LSTMModel` (`nn.Module`)**:
  * Red recurrente (`nn.LSTM` con `batch_first=True`), `nn.Dropout`, capa lineal (`nn.Linear`) y activación `nn.Sigmoid()` en el último paso temporal (`lstm_out[:, -1, :]`).
* **`RealTimeLSTMPredictor`**:
  * Carga `model_info.pkl`, los pesos del modelo (`.pth`) y los escaladores `scaler_x.pkl` y `scaler_y.pkl` en `cuda` (si está disponible) o `cpu`.
  * Valida que `scaler_x.n_features_in_` coincida con `model_info['input_dim']`.
  * `prepare_input_sequence()` construye el vector de entrada según `model_info['features']` (soportando alias como `temp_in`/`T_in`, `hum_in`/`H_in`, `temp_out`/`T_out`, `hum_out`/`H_out`, `temp_ref`/`T_ref`, `hum_ref`/`H_ref`, `error_temp` y `error_hum`).
  * `predict()` escala la secuencia con `scaler_x`, ejecuta la inferencia sin gradientes (`torch.no_grad()`) y desescala la salida con `scaler_y`.
* **`SerialCommunicator`**:
  * Gestiona la conexión serial (`baudrate=115200`, `timeout=2.0`, `STOPBITS_ONE`), reintentos (`max_retries=3`), limpieza de búfer, espera de respuesta con prefijo esperado (`cmd_A`, `cmd_B`, `cmd_V`, `cmd_C`, `cmd_H`, `cmd_S`), parseo de `cmd_S:...` e inicialización (`initialize_system`).
* **`ActuatorController`**:
  * Recorta (*clip*) y redondea las predicciones desescaladas `[window, heater, humidifier]` a los rangos físicos:
    * `window` (`V`): `0.0` a `25.0`
    * `heater` (`C`): `0.0` a `255.0`
    * `humidifier` (`H`): `0.0` a `127.0`
  * Genera comandos `V<int>`, `C<int>` y `H<int>`.
* **`apply_simple_control()`**:
  * Control de respaldo durante el calentamiento/llenado del historial (`len(state_sequence) < SEQUENCE_LENGTH`): activa `C50` si `temp_in < temp_ref - 2` (apaga con `C0` si `> temp_ref + 2`) y `H40` si `hum_in < hum_ref - 5` (apaga con `H0` si `> hum_ref + 5`).
* **`main()`**:
  * Inicializa predictor, conexión serial y consignas (`TEMP_REF = 26.0`, `HUM_REF = 70.0`).
  * Ejecuta ciclos de `PREDICTION_INTERVAL = 10` segundos leyendo `S0`, actualizando `state_sequence` (ventana deslizante de `10` pasos) y enviando comandos al ESP32 con pausa `POST_COMMAND_GAP_S` entre cada uno.
  * En `finally`, envía `V0`, `C0`, `H0` y cierra el puerto serial.

---

## Requisitos y artefactos necesarios

### Dependencias
* `torch`, `numpy`, `scikit-learn`, `joblib`, `pyserial`

### Archivos requeridos en el directorio de ejecución
* `lstm_model_trained.pth` (pesos del modelo)
* `model_info.pkl` (metadatos de arquitectura y `features`)
* `scaler_x.pkl` y `scaler_y.pkl` (escaladores ajustados en entrenamiento)

---

## Uso

1. Ajusta en `main()` las rutas de los artefactos, `SERIAL_PORT` (ej. `"COM6"` o `"/dev/ttyUSB0"`), `SEQUENCE_LENGTH`, `PREDICTION_INTERVAL`, `TEMP_REF` y `HUM_REF`.
2. Ejecuta el script:

```bash
python implement_test.py
