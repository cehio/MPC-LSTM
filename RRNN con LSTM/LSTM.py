
import random as _random
import joblib
import matplotlib
matplotlib.use('TkAgg')  # Cambiar a 'Qt5Agg' si usas PyQt5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
# -----------------------------
# Configuración de archivos y campos
# -----------------------------
RUTA_DATASET = "datos_simulacion_v5.csv"

# Features y Targets adaptados a la estructura exacta generada por MPC_AI.py
FEATURES = ['temp_ref', 'hum_ref', 'temp_in', 'hum_in', 'T_out', 'H_out',
            'error_temp', 'error_hum']
TARGETS = ['window', 'heater', 'humidifier']

# Rangos FÍSICOS de los actuadores definidos en la clase P de MPC_AI.py
RANGOS_TARGET = {
    'window':     (0.0,  25.0),   # Ángulo en grados (P.ANG_MIN a P.ANG_MAX)
    'heater':     (0.0, 255.0),   # PWM (0 a 255)
    'humidifier': (0.0, 127.0),   # PWM (0 a 127)

}

COL_GRUPO = 'scenario_group_id'
SEQ_LENGTH = 10
BATCH_SIZE = 64 #300
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
SEED = 42
SPLIT_SEED = 42
FRAC_VAL = 0.20
FRAC_TEST = 0.20

def split_estratificado(data, seed=SPLIT_SEED, frac_val=FRAC_VAL, frac_test=FRAC_TEST,
                        col_grupo=COL_GRUPO):
    if 'scenario_id' not in data.columns:
        # Si el CSV no incluye la columna scenario_id, se infiere por saltos temporales
        print("   [INFO] 'scenario_id' no encontrado en columnas. Asignando IDs por discontinuidad...")
        temp_ref_filled = data['temp_ref'].ffill().bfill()
        hum_ref_filled = data['hum_ref'].ffill().bfill()
        data['scenario_id'] = (temp_ref_filled.diff().ne(0) | hum_ref_filled.diff().ne(0)).cumsum()

    ids_reales = data['scenario_id'].unique().tolist()

    if col_grupo in data.columns:
        if (data.groupby('scenario_id')[col_grupo].nunique() > 1).any():
            raise ValueError(f"Hay scenario_id con más de un valor de '{col_grupo}'.")
        g_de = data.groupby('scenario_id')[col_grupo].first()
        unidades = g_de.unique().tolist()
        print(f"   Split por GRUPOS ('{col_grupo}'): {len(unidades)} grupos, {len(ids_reales)} episodios")
    else:
        g_de = None
        unidades = ids_reales
        print(f"   [AVISO] No existe la columna '{col_grupo}': el split es por scenario_id "
              f"(si hay variantes aumentadas, puede haber fuga entre splits).")

    rng = _random.Random(seed)
    unidades_shuffled = unidades.copy()
    rng.shuffle(unidades_shuffled)

    n = len(unidades_shuffled)
    n_test = max(1, round(n * frac_test)) if n >= 3 else 1
    n_val = max(1, round(n * frac_val)) if (n - n_test) >= 2 else 1

    test_u = sorted(unidades_shuffled[:n_test])
    val_u = sorted(unidades_shuffled[n_test:n_test + n_val])
    train_u = sorted(unidades_shuffled[n_test + n_val:])

    # CORREGIDO: antes se copiaba test a train en silencio (fuga); ahora es un error.
    if not train_u:
        raise ValueError(f"Con {n} unidades de reparto el split deja train vacío; "
                         f"se necesitan al menos 3 (añade escenarios).")

    if g_de is not None:
        a_ids = lambda gs: sorted(g_de.index[g_de.isin(gs)].tolist())
        train, val, test = a_ids(train_u), a_ids(val_u), a_ids(test_u)
        print(f"   Grupos    -> Train {len(train_u)} | Val {len(val_u)} | Test {len(test_u)}")
    else:
        train, val, test = train_u, val_u, test_u

    print(f"   Escenarios Totales ({len(ids_reales)}): {ids_reales}")
    print(f"   Train ({len(train)}): {train}")
    print(f"   Val   ({len(val)}): {val}")
    print(f"   Test  ({len(test)}): {test}")
    return train, val, test


def validar_splits(train_ids, val_ids, test_ids):
    solapados = ((set(train_ids) & set(val_ids))
                 | (set(train_ids) & set(test_ids))
                 | (set(val_ids) & set(test_ids)))
    if solapados:
        raise ValueError(f"Escenarios repetidos entre splits: {solapados}")

    for nombre, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        if not ids:
            raise ValueError(f"El split '{nombre}' quedó VACÍO.")


def validar_grupos(data, train_ids, val_ids, test_ids, col_grupo=COL_GRUPO):
    """CORREGIDO: ningún grupo (base + variantes aumentadas) puede estar en dos splits."""
    if col_grupo not in data.columns:
        return
    g_de = data.groupby('scenario_id')[col_grupo].first()
    gs = {nombre: set(g_de[ids]) for nombre, ids in
          [("train", train_ids), ("val", val_ids), ("test", test_ids)]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        comunes = gs[a] & gs[b]
        if comunes:
            raise ValueError(f"Fuga: los grupos {sorted(comunes)} aparecen en '{a}' y en '{b}'.")
    print("   Verificación OK: ningún grupo aparece en más de un split.")
def construir_scaler_y(targets, rangos=RANGOS_TARGET):
    faltantes = [t for t in targets if t not in rangos]
    if faltantes:
        raise ValueError(f"Falta definir el rango físico de: {faltantes} en RANGOS_TARGET")

    lo = np.array([[rangos[t][0] for t in targets]], dtype=float)
    hi = np.array([[rangos[t][1] for t in targets]], dtype=float)
    return MinMaxScaler(feature_range=(0, 1)).fit(np.vstack([lo, hi]))


def avisar_targets_fuera_de_rango(data, targets, rangos=RANGOS_TARGET):
    for t in targets:
        lo, hi = rangos[t]
        col = data[t]
        n_fuera = int(((col < lo) | (col > hi)).sum())
        if n_fuera:
            print(f"   [AVISO] '{t}': {n_fuera} valores fuera de [{lo}, {hi}] "
                  f"(min={col.min():.3f}, max={col.max():.3f})")
class TimeSeriesDataset(Dataset):
    def __init__(self, data, features, targets, seq_length=10,
                 scaler_x=None, scaler_y=None):
        if scaler_x is None or scaler_y is None:
            raise ValueError("TimeSeriesDataset requiere scaler_x y scaler_y ya ajustados")
        self.seq_length = seq_length
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.features = scaler_x.transform(data[features].values)
        self.targets = scaler_y.transform(data[targets].values)

    def __len__(self):
        return max(0, len(self.features) - self.seq_length + 1)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length - 1]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def construir_dataset_por_escenarios(data, escenarios, features, targets, seq_length,
                                     scaler_x, scaler_y, nombre=""):
    if not escenarios:
        raise ValueError(f"La lista de escenarios del split '{nombre}' está vacía.")

    subset = data[data['scenario_id'].isin(escenarios)]

    sub_datasets, descartados = [], []
    for sid in escenarios:
        g = subset[subset['scenario_id'] == sid].reset_index(drop=True)
        ds = TimeSeriesDataset(g, features, targets, seq_length=seq_length,
                               scaler_x=scaler_x, scaler_y=scaler_y)
        if len(ds) > 0:
            sub_datasets.append(ds)
        else:
            descartados.append((sid, len(g)))

    for sid, n in descartados:
        print(f"   [AVISO] '{sid}' descartado en '{nombre}': {n} filas < seq_length={seq_length}")

    if not sub_datasets:
        raise ValueError(f"El split '{nombre}' no produjo ninguna secuencia válida.")

    return ConcatDataset(sub_datasets)

class LSTMModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3,
                 learning_rate=1e-3, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        last_output = self.dropout(lstm_out[:, -1, :])
        return self.activation(self.fc(last_output))

    def _paso_comun(self, batch, nombre):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log(nombre, loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._paso_comun(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self._paso_comun(batch, 'val_loss')

    def test_step(self, batch, batch_idx):
        return self._paso_comun(batch, 'test_loss')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }

class TrainingPlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # CORREGIDO: el sanity check dispara este hook y dejaba val_losses con una época de más
        if trainer.sanity_checking:
            return
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

    def plot(self):
        plt.figure(figsize=(12, 5))
        for i, (escala, titulo) in enumerate([("linear", "Escala Lineal"), ("log", "Escala Logarítmica")]):
            plt.subplot(1, 2, i + 1)
            plt.plot(self.train_losses, label="Entrenamiento", marker="o", markersize=3)
            plt.plot(self.val_losses, label="Validación", marker="s", markersize=3)
            plt.yscale(escala)
            plt.xlabel("Época")
            plt.ylabel("Pérdida (MSE)")
            plt.title(f"Curvas de entrenamiento - {titulo}")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig("curvas_entrenamiento_v5.png", dpi=150)
        print("   Guardado: curvas_entrenamiento_v5.png")
        plt.close()

def _predecir_dataset(model, dataset):
    model.eval()
    device = next(model.parameters()).device
    preds, reals = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            pred = model(x.unsqueeze(0).to(device)).cpu().numpy().squeeze()
            preds.append(pred)
            reals.append(y.numpy())
    return np.array(preds), np.array(reals)


def evaluar_por_target(model, test_dataset, targets, scaler_y):
    preds, reals = _predecir_dataset(model, test_dataset)
    preds_real = scaler_y.inverse_transform(preds)
    reals_real = scaler_y.inverse_transform(reals)

    print(f"{'Target':<14} {'MAE (unid. real)':>18} {'MSE (unid. real)':>18} {'MAE (escalado 0-1)':>20}")
    print("-" * 72)
    for i, t in enumerate(targets):
        mae_real = np.mean(np.abs(preds_real[:, i] - reals_real[:, i]))
        mse_real = np.mean((preds_real[:, i] - reals_real[:, i]) ** 2)
        mae_scaled = np.mean(np.abs(preds[:, i] - reals[:, i]))
        print(f"{t:<14} {mae_real:>18.3f} {mse_real:>18.3f} {mae_scaled:>20.4f}")


def graficar_escenario(model, data, scenario_id, features, targets, seq_length,
                       scaler_x, scaler_y, guardar=None):
    g = data[data['scenario_id'] == scenario_id].reset_index(drop=True)
    ds = TimeSeriesDataset(g, features, targets, seq_length=seq_length,
                           scaler_x=scaler_x, scaler_y=scaler_y)
    if len(ds) == 0:
        print(f"   Escenario {scenario_id}: muy corto para graficar (menos de {seq_length} pasos)")
        return

    preds, reals = _predecir_dataset(model, ds)
    preds_real = scaler_y.inverse_transform(preds)
    reals_real = scaler_y.inverse_transform(reals)
    t_axis = np.arange(len(preds_real))

    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 3 * len(targets)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, t in enumerate(targets):
        axes[i].plot(t_axis, reals_real[:, i], label='Real MPC', linewidth=1.5)
        axes[i].plot(t_axis, preds_real[:, i], label='Predicción LSTM', linewidth=1.2, linestyle='--')
        axes[i].set_ylabel(t)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.4)
    axes[-1].set_xlabel('Pasos de simulación')
    fig.suptitle(f'Predicción vs Real — Escenario: {scenario_id}')
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=150)
        print(f"   Guardado: {guardar}")
    plt.close()

def main():
    pl.seed_everything(SEED, workers=True)

    print("Cargando datos...")
    data = pd.read_csv(RUTA_DATASET)

    if data.empty:
        raise ValueError("El archivo de datos está vacío o no se pudo cargar.")

    features, targets = FEATURES, TARGETS

    requeridas = features + targets
    faltantes = [c for c in requeridas if c not in data.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")

    print("Verificando rangos físicos de los targets...")
    avisar_targets_fuera_de_rango(data, targets)

    print("Ejecutando split por escenario...")
    train_ids, val_ids, test_ids = split_estratificado(data)
    validar_splits(train_ids, val_ids, test_ids)
    validar_grupos(data, train_ids, val_ids, test_ids)   # CORREGIDO

    # Scalers
    scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(
        data[data['scenario_id'].isin(train_ids)][features].values
    )
    scaler_y = construir_scaler_y(targets)

    print("Construyendo datasets...")
    train_dataset = construir_dataset_por_escenarios(
        data, train_ids, features, targets, SEQ_LENGTH, scaler_x, scaler_y, "train"
    )
    val_dataset = construir_dataset_por_escenarios(
        data, val_ids, features, targets, SEQ_LENGTH, scaler_x, scaler_y, "val"
    )
    test_dataset = construir_dataset_por_escenarios(
        data, test_ids, features, targets, SEQ_LENGTH, scaler_x, scaler_y, "test"
    )

    print(f"Tamaños de las secuencias -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    effective_batch_size = max(1, min(BATCH_SIZE, len(train_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        num_workers=4,
        pin_memory=True
    )

    model = LSTMModel(
        input_dim=len(features),
        output_dim=len(targets),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
    )

    # Callbacks
    plot_callback = TrainingPlotCallback()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min",
        min_delta=1e-5,
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="checkpoints_v5/",
        filename="best_lstm_v5-{epoch:02d}-{val_loss:.5f}",
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stopping, plot_callback, checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        accelerator='auto',
        devices=1,
        deterministic=True,
    )

    print("\nIniciando entrenamiento...")
    trainer.fit(model, train_loader, val_loader)

    print("\nEvaluando el mejor modelo en el conjunto de prueba (Test)...")
    test_result = trainer.test(model, test_loader, ckpt_path="best")
    print(f"\nPérdida MSE final en Test: {test_result[0]['test_loss']:.6f}")

    # Cargar los pesos del mejor checkpoint para inferencias y guardado manual
    print(f"\nCargando pesos del mejor checkpoint: {checkpoint_callback.best_model_path}")
    best_model = LSTMModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        input_dim=len(features),
        output_dim=len(targets),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
    )
    best_model.eval()

    # Guardar tanto en versiones identificadas _v5 como en los nombres estándar que carga Implement_test.py
    print("\nGuardando modelo y artefactos (v5 y estándar)...")
    torch.save(best_model.state_dict(), "lstm_model_trained.pth")
    torch.save(best_model.state_dict(), "lstm_model_trained_v5.pth")
    joblib.dump(scaler_x, "scaler_x.pkl")
    joblib.dump(scaler_x, "scaler_x_v5.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    joblib.dump(scaler_y, "scaler_y_v5.pkl")

    model_info = {
        'input_dim': len(features),
        'output_dim': len(targets),
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'features': features,
        'targets': targets,
        'seq_length': SEQ_LENGTH,
        'rangos_target': RANGOS_TARGET,
        'salida_sigmoid': True,
        'escenarios_train': train_ids,
        'escenarios_val': val_ids,
        'escenarios_test': test_ids,
        'best_checkpoint_path': checkpoint_callback.best_model_path,
    }
    joblib.dump(model_info, "model_info.pkl")
    joblib.dump(model_info, "model_info_v5.pkl")
    print("✓ Modelo, scalers e información guardados con éxito en v5 y estándar.")

    plot_callback.plot()

    print("\nEvaluación detallada por variable objetivo:")
    evaluar_por_target(best_model, test_dataset, targets, scaler_y)

    print("\nGenerando gráficas para escenarios de prueba...")
    for sid in test_ids[:2]:
        graficar_escenario(
            best_model, data, sid, features, targets, SEQ_LENGTH,
            scaler_x, scaler_y, guardar=f"prediccion_escenario_v5_{sid}.png"
        )

if __name__ == "__main__":
    main()
