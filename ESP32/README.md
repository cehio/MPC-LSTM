# Firmware ESP32 - Monitoreo y Control

Firmware para **ESP32** (Arduino / ESP32 core) encargado de leer dos sensores **AHT10** (interior y exterior), controlar actuadores (ventana con servomotor, calentador y humidificador por PWM), servir una página web local de monitoreo y enviar telemetría periódica por **HTTP POST** a un servidor remoto. Además expone un protocolo por **puerto serial** compatible con scripts de control (como el controlador de histéresis en Python).

---

## Hardware y pines

| Componente | Pin / Dirección / Bus | Notas |
| :--- | :--- | :--- |
| Sensor interior (`aht10_in`) | `SDA_1 = 21`, `SCL_1 = 22` (`0x38`) | 100 kHz |
| Sensor exterior (`aht10_out`) | `SDA_2 = 25`, `SCL_2 = 26` (`0x39`) | 100 kHz |
| Servomotor ventana (`VentServo`) | GPIO `15` (`SM`) | Rango `0°`–`90°`; se desacopla tras `1500 ms` de inactividad |
| Calentador (`HeaterPin`) | GPIO `12` (`PWM`, `0..255`) | Controlado mediante comando `C` |
| Humidificador (`HumPin`) | GPIO `17` (`PWM`, `0..127` en `processCommand`) | Frecuencia `111000 Hz`, resolución `8` bits |
| LED de estado (`LED`) | GPIO `19` | Alterna estado en la interrupción del temporizador (`10 s`) |
| Pin PWM auxiliar (`heater_pin`) | GPIO `13` (`PWM`, `1000 Hz`, `8` bits) | Inicializado en `0` en `setup()` |
| Pin PWM auxiliar (`ledPin`) | GPIO `16` (`PWM`, `111000 Hz`, `8` bits) | Inicializado en `0` en `setup()` |

---

## Funcionalidades principales

* **Filtrado de señales con Kalman 1D (`KalmanFilter1D`)**:
  * Filtra `temp_in`, `temp_out`, `hum_in` y `hum_out` ignorando lecturas `NaN` e inicializando el estado con la primera medición válida.
* **Control de actuadores**:
  * **Ventana (`V`)**: ajusta el ángulo entre `0` y `90` grados, acopla el servo si estaba desacoplado y reinicia `servoTimer` (`1500 ms`) para liberar el pin (`detach()`) tras el tiempo de inactividad.
  * **Calentador (`C`)**: limita el valor a `0..255` y escribe en `HeaterPin` (`ledcWrite`).
  * **Humidificador (`H`)**: limita el valor a `0..127` y escribe en `HumPin` (`ledcWrite`).
* **Servidor web local (Puerto `80`)**:
  * Sirve una página HTML con auto-refresco cada `10` segundos (`<meta http-equiv="refresh" content="10">`) mostrando temperatura y humedad (interior/exterior) y el estado de ventana, calentador y humidificador.
* **Envío periódico a servidor (`sendDataToServer`)**:
  * Un temporizador hardware (`1 MHz` con alarma cada `10,000,000` ticks = `10 s`) activa `shouldSendRequest = true`.
  * Envía por `POST` (`application/x-www-form-urlencoded`) los campos `api_key` y `value1` a `value9` (temperaturas y humedades filtradas, `window`, `heater`, `humifier`, `ref_temp`, `ref_hum`) con *timeout* de `15000 ms`.

---

## Protocolo por puerto serial (`115200` baudios)

Los comandos se leen con `Serial.readString()` y se recortan con `.trim()`. Si el formato es inválido o falta valor numérico, responde `error:invalid_format` o `error:no_value`; si el comando no existe, responde `error:unknown_cmd`.

| Comando | Ejemplo | Respuesta | Acción |
| :--- | :--- | :--- | :--- |
| `V<int>` | `V45` | `cmd_V:45` | Mueve la ventana (`0..90`) |
| `C<int>` | `C255` | `cmd_C:255` | Ajusta el calentador (`0..255`) |
| `H<int>` | `H127` | `cmd_H:127` | Ajusta el humidificador (`0..127`) |
| `A<float>` | `A26.0` | `cmd_A:26.00` | Actualiza `ref_temp` |
| `B<float>` | `B70.0` | `cmd_B:70.00` | Actualiza `ref_hum` |
| `T...` | `T0` | `cmd_T:<val>` | Lee temperatura interior filtrada |
| `Y...` | `Y0` | `cmd_Y:<val>` | Lee temperatura exterior filtrada |
| `I...` | `I0` | `cmd_I:<val>` | Lee humedad interior filtrada |
| `J...` | `J0` | `cmd_J:<val>` | Lee humedad exterior filtrada |
| `S...` | `S0` | `cmd_S:temp_in:...,temp_out:...,hum_in:...,hum_out:...` | Lee los 4 valores de sensores |
| `R...` | `R0` | `cmd_R:ref_temp:...,ref_hum:...` | Lee referencias actuales |
| `D...` | `D0` | `cmd_D:window:...,heater:...,humidifier:...` | Lee estado actual de actuadores |

---

## Configuración y dependencias

### Librerías de Arduino requeridas
* `WiFi.h` y `HTTPClient.h` (incluidas en el soporte de ESP32 para Arduino)
* `Wire.h`
* `AHTxx` (para sensores AHT10/AHT20)
* `ESP32Servo`

### Antes de compilar
1. Reemplaza las credenciales y URL marcadas con `*` (`ssid`, `password`, `serverName`, `apiKeyValue`).
2. Verifica que los pines PWM y los límites de actuadores coincidan con tu etapa de potencia física.
3. Revisa y limpia los elementos sin usar señalados en la sección anterior (`I2C_1`, `I2C_2`, `ledPin`, `heater_pin` y la variable `success`) si deseas mantener el código más claro.
