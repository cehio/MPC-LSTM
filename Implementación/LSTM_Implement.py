import torch
import torch.nn as nn
import time
import serial
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Dict, List, Optional
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        last_output = self.dropout(lstm_out[:, -1, :])
        return self.activation(self.fc(last_output))
class RealTimeLSTMPredictor:
    def __init__(self, model_weights_path, model_info_path, sequence_length=10,
                 scaler_x_path="scaler_x.pkl", scaler_y_path="scaler_y.pkl"):
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        try:
            self.model_info = joblib.load(model_info_path)
            logger.info(f"Model info loaded: {self.model_info}")
            self.model = LSTMModel(
                input_dim=self.model_info['input_dim'],
                output_dim=self.model_info['output_dim'],
                hidden_dim=self.model_info['hidden_dim'],
                num_layers=self.model_info['num_layers'],
                dropout=self.model_info.get('dropout', 0.3)
            )

            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.scaler_x = joblib.load(scaler_x_path)
            self.scaler_y = joblib.load(scaler_y_path)

            n_esperado = self.scaler_x.n_features_in_
            if n_esperado != self.model_info['input_dim']:
                raise ValueError(
                    f"scaler_x espera {n_esperado} features pero model_info declara "
                    f"input_dim={self.model_info['input_dim']}: scaler_x.pkl y model_info.pkl "
                    f"no son del mismo entrenamiento. Revisar con que version del script de "
                    f"entrenamiento se genero cada archivo."
                )

            logger.info("Modelo cargado correctamente")
            logger.info(f"Arquitectura: input_dim={self.model_info['input_dim']}, "
                        f"output_dim={self.model_info['output_dim']}, "
                        f"hidden_dim={self.model_info['hidden_dim']}, "
                        f"num_layers={self.model_info['num_layers']}")

        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}")
            raise

    def prepare_input_sequence(self, sensor_data, temp_ref=25.0, hum_ref=70.0, error_temp=0.0, error_hum=0.0):

        try:
            valores_disponibles = {
                'temp_ref': temp_ref, 'T_ref': temp_ref,
                'hum_ref': hum_ref, 'H_ref': hum_ref,
                'temp_in': sensor_data.get('temp_in', temp_ref),
                'T_in': sensor_data.get('temp_in', temp_ref),
                'hum_in': sensor_data.get('hum_in', hum_ref),
                'H_in': sensor_data.get('hum_in', hum_ref),
                'T_out': sensor_data.get('temp_out', 25.0),
                'temp_out': sensor_data.get('temp_out', 25.0),
                'H_out': sensor_data.get('hum_out', 60.0),
                'hum_out': sensor_data.get('hum_out', 60.0),
                'error_temp': error_temp,
                'error_hum': error_hum,
            }

            columnas = self.model_info['features']
            faltantes = [c for c in columnas if c not in valores_disponibles]
            if faltantes:
                raise KeyError(
                    f"model_info['features'] pide columnas que prepare_input_sequence "
                    f"no sabe calcular: {faltantes}"
                )

            input_array = np.array([[valores_disponibles[c] for c in columnas]])
            return input_array

        except Exception as e:
            logger.error(f"Error preparando secuencia de entrada: {e}")
            return None

    def predict(self, input_sequence):

        if self.model is None:
            logger.error("Modelo no cargado")
            return None

        try:
            if input_sequence.ndim == 2:
                input_sequence = input_sequence.reshape(1, self.sequence_length, -1)
            original_shape = input_sequence.shape
            input_reshaped = input_sequence.reshape(-1, original_shape[2])
            input_scaled = self.scaler_x.transform(input_reshaped)
            input_scaled = input_scaled.reshape(original_shape)
            input_tensor = torch.FloatTensor(input_scaled).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction_np = prediction.cpu().numpy()
            prediction_descaled = self.scaler_y.inverse_transform(prediction_np)
            logger.info(f"Predicción desescalada: {prediction_descaled[0]}")
            return prediction_descaled[0]
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


class SerialCommunicator:

    PRE_RESPONSE_SLEEP_S = 0.2
    POST_COMMAND_GAP_S = 0.1
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.last_response = ""

    def connect(self) -> bool:
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                stopbits=serial.STOPBITS_ONE
            )
            logger.info(f"Conexión establecida en {self.port}")
            return True
        except Exception as e:
            logger.error(f"Error conectando a {self.port}: {e}")
            return False

    def disconnect(self) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Conexión serial cerrada")

    def send_command(self, command: str, expected_prefix: str = "cmd", max_retries: int = 3) -> bool:
        clean_command = command.strip().upper().replace(" ", "") + '\n'
        self.last_response = ""

        for attempt in range(max_retries):
            try:
                if not self.serial_conn or not self.serial_conn.is_open:
                    logger.error("Conexión serial no disponible")
                    return False

                self.serial_conn.reset_input_buffer()
                self.serial_conn.write(clean_command.encode('utf-8'))
                logger.info(f"Comando enviado: {clean_command.strip()}")

                time.sleep(self.PRE_RESPONSE_SLEEP_S)
                start_time = time.time()

                while time.time() - start_time < 3.0:
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            self.last_response = line
                            logger.info(f"Respuesta: {line}")

                            if line.startswith(expected_prefix):
                                logger.info(f"Comando procesado: {line}")
                                return True
                            elif line.startswith("error"):
                                logger.error(f"Error del ESP32: {line}")
                                return False
                    time.sleep(0.1)

                logger.warning(f"Timeout esperando respuesta para: {clean_command.strip()}")

            except Exception as e:
                logger.error(f"Intento {attempt + 1} fallido: {e}")

        logger.error(f"Fallo al enviar comando después de {max_retries} intentos: {command}")
        return False

    def get_last_response(self) -> str:
        return self.last_response

    def parse_sensor_data(self, response: str) -> dict:
        if not response.startswith("cmd_S:"):
            return {}

        try:
            data_str = response[6:]
            data_dict = {}
            pairs = data_str.split(',')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    data_dict[key.strip()] = float(value.strip())
            return data_dict

        except Exception as e:
            logger.error(f"Error parseando datos sensores: {e}")
            return {}

    def initialize_system(self, temp_ref=25.0, hum_ref=70.0):
        logger.info(f"Inicializando sistema - Temp_ref: {temp_ref}°C, Hum_ref: {hum_ref}%")
        commands = [
            (f"A{temp_ref}", "cmd_A"),
            (f"B{hum_ref}", "cmd_B"),
            ("V0", "cmd_V"),
            ("C0", "cmd_C"),
            ("H0", "cmd_H")
        ]
        for cmd, expected_prefix in commands:
            if not self.send_command(cmd, expected_prefix):
                logger.error(f"Fallo en inicialización con comando: {cmd}")
                return False

        logger.info("Sistema inicializado correctamente")
        return True

class ActuatorController:
    def __init__(self):
        # Rangos sincronizados con RANGOS_TARGET de tu script de entrenamiento
        self.limits = {
            'window': (0.0, 25.0),
            'heater': (0.0, 255.0),
            'humidifier': (0.0, 127.0),
        }

    def format_prediction_to_commands(self, prediction_real_units: np.ndarray) -> Dict[str, str]:
        if prediction_real_units is None or len(prediction_real_units) != 3:
            return {}

        try:
            window_val, heater_val, humidifier_val = prediction_real_units
            window_value = int(round(np.clip(window_val, self.limits['window'][0], self.limits['window'][1])))
            heater_value = int(round(np.clip(heater_val, self.limits['heater'][0], self.limits['heater'][1])))
            humidifier_value = int(round(np.clip(humidifier_val, self.limits['humidifier'][0], self.limits['humidifier'][1])))
            commands = {
                'window': f"V{window_value}",
                'heater': f"C{heater_value}",
                'humidifier': f"H{humidifier_value}"
            }

            logger.info(f"Comandos formateados: V={window_value}°, C={heater_value}, H={humidifier_value}")
            return commands

        except Exception as e:
            logger.error(f"Error formateando predicciones: {e}")
            return {}


def apply_simple_control(serial_comm, sensor_data, temp_ref, hum_ref):
    temp_in = sensor_data.get('temp_in', temp_ref)
    hum_in = sensor_data.get('hum_in', hum_ref)

    if temp_in < temp_ref - 2:
        serial_comm.send_command("C50", "cmd_C")
        logger.info(f"Control simple: Calentador ON (temp_in={temp_in:.1f}°C < ref={temp_ref}°C)")
    elif temp_in > temp_ref + 2:
        serial_comm.send_command("C0", "cmd_C")
        logger.info(f"Control simple: Calentador OFF (temp_in={temp_in:.1f}°C > ref={temp_ref}°C)")

    if hum_in < hum_ref - 5:
        serial_comm.send_command("H40", "cmd_H")
        logger.info(f"Control simple: Humidificador ON (hum_in={hum_in:.1f}% < ref={hum_ref}%)")
    elif hum_in > hum_ref + 5:
        serial_comm.send_command("H0", "cmd_H")
        logger.info(f"Control simple: Humidificador OFF (hum_in={hum_in:.1f}% > ref={hum_ref}%)")


def main():
    MODEL_WEIGHTS_PATH = "lstm_model_trained.pth"
    MODEL_INFO_PATH = "model_info.pkl"
    SCALER_X_PATH = "scaler_x.pkl"
    SCALER_Y_PATH = "scaler_y.pkl"
    SERIAL_PORT = "COM6"
    SEQUENCE_LENGTH = 10
    PREDICTION_INTERVAL = 10

    TEMP_REF = 26.0
    HUM_REF = 70.0

    try:
        predictor = RealTimeLSTMPredictor(
            model_weights_path=MODEL_WEIGHTS_PATH,
            model_info_path=MODEL_INFO_PATH,
            sequence_length=SEQUENCE_LENGTH,
            scaler_x_path=SCALER_X_PATH,
            scaler_y_path=SCALER_Y_PATH
        )
        logger.info("Predictor LSTM inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar el predictor: {e}")
        return

    serial_comm = SerialCommunicator(SERIAL_PORT)
    actuator_controller = ActuatorController()

    if not serial_comm.connect():
        logger.error("No se pudo conectar al puerto serial")
        return

    if not serial_comm.initialize_system(temp_ref=TEMP_REF, hum_ref=HUM_REF):
        logger.error("Fallo en inicialización del sistema")
        return

    logger.info("Sistema listo. Iniciando bucle principal...")
    state_sequence = []

    try:
        while True:
            loop_start = time.perf_counter()

            if serial_comm.send_command("S0", "cmd_S"):
                response = serial_comm.get_last_response()
                sensor_data = serial_comm.parse_sensor_data(response)

                if sensor_data:
                    logger.info(f"📊 Datos sensores: {sensor_data}")
                    error_temp = TEMP_REF - sensor_data.get('temp_in', TEMP_REF)
                    error_hum = HUM_REF - sensor_data.get('hum_in', HUM_REF)
                    current_state = predictor.prepare_input_sequence(
                        sensor_data, TEMP_REF, HUM_REF, error_temp, error_hum
                    )
                    if current_state is not None:
                        state_sequence.append(current_state[0])
                        if len(state_sequence) > SEQUENCE_LENGTH:
                            state_sequence = state_sequence[-SEQUENCE_LENGTH:]
                        if len(state_sequence) == SEQUENCE_LENGTH:
                            sequence_array = np.array(state_sequence)
                            predicted_actions = predictor.predict(sequence_array)
                            if predicted_actions is not None:
                                commands = actuator_controller.format_prediction_to_commands(predicted_actions)
                                prefix_map = {
                                    'window': 'cmd_V',
                                    'heater': 'cmd_C',
                                    'humidifier': 'cmd_H'
                                }
                                for actuator, cmd in commands.items():
                                    expected_pfx = prefix_map.get(actuator, "cmd")
                                    if serial_comm.send_command(cmd, expected_pfx):
                                        logger.info(f"Comando {cmd} enviado exitosamente")
                                    else:
                                        logger.error(f"Fallo al enviar comando {cmd}")
                                    time.sleep(serial_comm.POST_COMMAND_GAP_S)

                        else:
                            logger.info(f"📈 Recolectando datos históricos... ({len(state_sequence)}/{SEQUENCE_LENGTH})")
                            apply_simple_control(serial_comm, sensor_data, TEMP_REF, HUM_REF)
            elapsed = time.perf_counter() - loop_start
            sleep_time = PREDICTION_INTERVAL - elapsed
            if sleep_time > 0:
                logger.info(f"Ciclo tardo {elapsed:.2f}s, durmiendo {sleep_time:.2f}s "
                           f"para completar el periodo de {PREDICTION_INTERVAL}s")
                time.sleep(sleep_time)
            else:
                logger.warning(f"Ciclo tardo {elapsed:.2f}s, MAS que el periodo objetivo "
                               f"de {PREDICTION_INTERVAL}s -- no se pudo mantener el paso fijo. "
                               f"Revisar latencia serial (send_command) si esto se repite.")
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado. Cerrando...")
    except Exception as e:
        logger.error(f"Error en bucle principal: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("🔌 Apagando actuadores...")
        serial_comm.send_command("V0", "cmd_V")
        serial_comm.send_command("C0", "cmd_C")
        serial_comm.send_command("H0", "cmd_H")
        serial_comm.disconnect()
        logger.info("Aplicación terminada correctamente")


if __name__ == "__main__":
    main()
