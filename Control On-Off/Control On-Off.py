
import time
import serial
import numpy as np
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HysteresisController:
    def __init__(self, temp_hysteresis=1.0, hum_hysteresis=3.0):
        self.temp_hyst = temp_hysteresis
        self.hum_hyst = hum_hysteresis
        self.heater_on = False
        self.humidifier_on = False

    def apply_control(self, serial_comm, sensor_data, temp_ref, hum_ref):
        temp_in = sensor_data.get('temp_in', temp_ref)
        hum_in = sensor_data.get('hum_in', hum_ref)
        if not self.heater_on and temp_in < (temp_ref - self.temp_hyst):
            if serial_comm.send_command("C255", "cmd_C"):
                self.heater_on = True
                logger.info(f"Calentador ON: temp_in ({temp_in:.1f}°C) < ref - hyst ({temp_ref - self.temp_hyst}°C)")

        elif self.heater_on and temp_in > (temp_ref + self.temp_hyst):
            if serial_comm.send_command("C0", "cmd_C"):
                self.heater_on = False
                logger.info(
                    f"Calentador OFF: temp_in ({temp_in:.1f}°C) > ref + hyst ({temp_ref + self.temp_hyst}°C)")

        time.sleep(serial_comm.POST_COMMAND_GAP_S)

        if not self.humidifier_on and hum_in < (hum_ref - self.hum_hyst):
            if serial_comm.send_command("H127", "cmd_H"):
                self.humidifier_on = True
                logger.info(f"Humidificador ON: hum_in ({hum_in:.1f}%) < ref - hyst ({hum_ref - self.hum_hyst}%)")

        elif self.humidifier_on and hum_in > (hum_ref + self.hum_hyst):
            if serial_comm.send_command("H0", "cmd_H"):
                self.humidifier_on = False
                logger.info(f"Humidificador OFF: hum_in ({hum_in:.1f}%) > ref + hyst ({hum_ref + self.hum_hyst}%)")
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
            logger.info("🔌 Conexión serial cerrada")

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
        logger.info(f"🔧 Inicializando sistema - Temp_ref: {temp_ref}°C, Hum_ref: {hum_ref}%")
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
            logger.info(f"🎛Comandos formateados: V={window_value}°, C={heater_value}, H={humidifier_value}")
            return commands
        except Exception as e:
            logger.error(f"Error formateando predicciones: {e}")
            return {}
        
def main():
    SERIAL_PORT = "COM6"
    PREDICTION_INTERVAL = 10
    TEMP_REF = 26.0
    HUM_REF = 70.0
    serial_comm = SerialCommunicator(SERIAL_PORT)
    hysteresis_controller = HysteresisController(temp_hysteresis=1.0, hum_hysteresis=3.0)
    if not serial_comm.connect():
        logger.error("No se pudo conectar al puerto serial")
        return

    if not serial_comm.initialize_system(temp_ref=TEMP_REF, hum_ref=HUM_REF):
        logger.error("Fallo en inicialización del sistema")
        return
    logger.info("Sistema listo. Iniciando bucle principal...")
    try:
        while True:
            loop_start = time.perf_counter()

            if serial_comm.send_command("S0", "cmd_S"):
                response = serial_comm.get_last_response()
                sensor_data = serial_comm.parse_sensor_data(response)

                if sensor_data:
                    logger.info(f"Datos sensores: {sensor_data}")

                    hysteresis_controller.apply_control(serial_comm, sensor_data, TEMP_REF, HUM_REF)


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
        logger.info("Apagando actuadores...")
        serial_comm.send_command("V0", "cmd_V")
        serial_comm.send_command("C0", "cmd_C")
        serial_comm.send_command("H0", "cmd_H")
        serial_comm.disconnect()
        logger.info("Aplicación terminada correctamente")


if __name__ == "__main__":
    main()
