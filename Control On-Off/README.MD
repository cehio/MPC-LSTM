# Controlador On-Off por Puerto Serial (ESP32)

Script en Python para monitorear sensores de temperatura y humedad y controlar actuadores (calentador, humidificador y ventana) a través de una conexión serial con un microcontrolador **ESP32**. Incluye control por histéresis, reintentos de comunicación con *timeout* y formateo de salidas para modelos de predicción (`ActuatorController`).

---

## Características principales

* **Comunicación serial robusta (`SerialCommunicator`)**:
  * Conexión configurable (`port`, `baudrate=115200`, `timeout=2.0`, 1 bit de parada).
  * Envío de comandos normalizados (`strip().upper().replace(" ", "") + '\n'`), limpieza del búfer de entrada y espera de respuesta con *prefijo esperado* (`cmd_A`, `cmd_B`, `cmd_V`, `cmd_C`, `cmd_H`, `cmd_S`) o detección de `error`.
  * Hasta `max_retries=3` intentos por comando y ventana de espera de hasta `3.0s` por intento.
  * Parser para tramas de sensores con formato `cmd_S:clave1:valor1,clave2:valor2,...` (ej. `temp_in`, `hum_in`).
* **Control por histéresis (`HysteresisController`)**:
  * Evita el encendido/apagado constante (*chattering*) alrededor de los valores de referencia (`temp_ref` y `hum_ref`) usando bandas de histéresis (`temp_hysteresis=1.0`, `hum_hysteresis=3.0`).
  * Activa el calentador (`C255`) cuando `temp_in < temp_ref - temp_hyst` y lo apaga (`C0`) cuando `temp_in > temp_ref + temp_hyst`.
  * Activa el humidificador (`H127`) cuando `hum_in < hum_ref - hum_hyst` y lo apaga (`H0`) cuando `hum_in > hum_ref + hum_hyst`.
  * Respeta una pausa entre comandos (`serial_comm.POST_COMMAND_GAP_S`).
* **Formateo de actuadores (`ActuatorController`)**:
  * Convierte un arreglo NumPy de 3 elementos `[window, heater, humidifier]` en unidades reales a comandos seriales (`V...`, `C...`, `H...`), recortando (*clipping*) y redondeando a enteros dentro de los límites:
    * `window` (`V`): `0.0` a `25.0` (grados)
    * `heater` (`C`): `0.0` a `255.0`
    * `humidifier` (`H`): `0.0` a `127.0`
* **Bucle principal de periodo fijo (`main`)**:
  * Inicializa referencias y apaga actuadores al arrancar.
  * Lee sensores (`S0`), aplica control por histéresis y ajusta el tiempo de espera para mantener un ciclo objetivo de `10` segundos (`PREDICTION_INTERVAL`), registrando advertencias si el ciclo excede el tiempo.
  * En su bloque `finally` (ante `KeyboardInterrupt`, excepciones o cierre), envía `V0`, `C0` y `H0` antes de cerrar el puerto serial.

---

## Requisitos

* Python 3.8+
* Dependencias:
  * `pyserial` (`import serial`)
  * `numpy`

Instalación:

```bash
pip install pyserial numpy
