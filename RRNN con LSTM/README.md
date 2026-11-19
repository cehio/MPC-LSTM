# Entrenamiento de Modelo LSTM en base a dataset del MPC

Pipeline en **PyTorch Lightning** para entrenar, validar y evaluar una red **LSTM** que aprende a predecir las acciones de control (`window`, `heater`, `humidifier`) a partir de series temporales de variables ambientales y errores de referencia generadas por simulación/MPC (`datos_simulacion_v5.csv`).

---

## Características principales

* **División libre de fugas por escenario y grupo (`split_estratificado`, `validar_splits`, `validar_grupos`)**:
  * Reparte episodios (`scenario_id`) o grupos (`scenario_group_id`) entre **Train**, **Validación** y **Test** (`FRAC_VAL = 0.20`, `FRAC_TEST = 0.20`), impidiendo que variantes de un mismo grupo queden en distintos conjuntos o que algún split quede vacío.
* **Ventanas temporales por episodio (`TimeSeriesDataset`, `construir_dataset_por_escenarios`)**:
  * Construye secuencias de longitud `SEQ_LENGTH = 10` dentro de cada `scenario_id` (sin cruzar fronteras entre episodios) y descarta episodios más cortos que `SEQ_LENGTH`.
* **Escalado controlado**:
  * `scaler_x`: `MinMaxScaler(feature_range=(0, 1))` ajustado **exclusivamente** sobre los datos de entrenamiento (`train_ids`).
  * `scaler_y`: `MinMaxScaler(feature_range=(0, 1))` ajustado sobre los rangos físicos definidos en `RANGOS_TARGET` (`window`: `0..25`, `heater`: `0..255`, `humidifier`: `0..127`).
* **Arquitectura y entrenamiento (`LSTMModel`)**:
  * `nn.LSTM` (`input_dim=8`, `hidden_dim=128`, `num_layers=3`, `dropout=0.3`), `nn.Dropout`, `nn.Linear(hidden_dim, 3)` y activación `nn.Sigmoid()` (salida en `[0, 1]`).
  * Pérdida **MSE**, optimizador **Adam** (`lr=1e-3`, `weight_decay=1e-5`), *scheduler* `ReduceLROnPlateau` monitoreando `val_loss`, *gradient clipping* (`1.0`), `EarlyStopping` (`patience=15`) y `ModelCheckpoint` (`save_top_k=1`).
* **Evaluación y exportación de artefactos**:
  * Evalúa el mejor *checkpoint* en `test_loader` y calcula **MAE** y **MSE** en unidades reales y en escala `[0, 1]` por cada *target* (`evaluar_por_target`).
  * Guarda pesos (`lstm_model_trained.pth` y `lstm_model_trained_v5.pth`), scalers (`scaler_x*.pkl`, `scaler_y*.pkl`), metadatos (`model_info*.pkl`) y gráficas (`curvas_entrenamiento_v5.png` y `prediccion_escenario_v5_<id>.png`).

---

## Datos de entrada esperados

Archivo CSV configurable en `RUTA_DATASET` (por defecto `datos_simulacion_v5.csv`) con las siguientes columnas:

* **Features (`FEATURES`)**: `temp_ref`, `hum_ref`, `temp_in`, `hum_in`, `T_out`, `H_out`, `error_temp`, `error_hum`
* **Targets (`TARGETS`)**: `window`, `heater`, `humidifier`
* **Identificadores (opcionales/recomendados)**: `scenario_id` y `scenario_group_id`

---

## Instalación y ejecución

### Dependencias principales
* `torch`, `pytorch-lightning`
* `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`

### Ejecución

```bash
python train_lstm_v5.py
