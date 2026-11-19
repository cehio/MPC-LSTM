#include <WiFi.h>
#include <HTTPClient.h>
#include <AHTxx.h>
#include <Wire.h>
#include <ESP32Servo.h>
AHTxx aht10_in(0x38, AHT1x_SENSOR);  // Sensor interior en 0x38
AHTxx aht10_out(0x39, AHT1x_SENSOR); // Sensor exterior en 0x39
#define SDA_1 21
#define SCL_1 22
#define SDA_2 25
#define SCL_2 26

#define SM        15  // verde (Servo motor)
#define HeaterPin 12  // rojo
#define HumPin    17  // amarillo
#define LED       19

const char* ssid = "*********";
const char* password = "*********";
const char* serverName = "http://monitoreocontrol.atwebpages.com/post-****.php";
String apiKeyValue = "******";

WiFiServer server(80);
String header;
String datain;

Servo VentServo;
bool servoAttached = false;
unsigned long servoTimer = 0;
const unsigned long servoTimeout = 1500;

// Configuración PWM humifier
const int ledPin = 16;
const int freq = 111000;
const int resolution = 8;
const int dutyCycle = 0;

const int heater_pin = 13;
const int freq2 = 1000;
const int resolution2 = 8;
const int dutyCycle2 = 0;

hw_timer_t *timer = NULL;

int window = 0;
int heater = 0;
int humifier = 0;

float ref_temp = 20;
float ref_hum = 70;

bool shouldSendRequest = false;
HTTPClient http;
String httpRequestData;

struct KalmanFilter1D {
  float Q; 
  float R; 
  float P; 
  float x; 
  bool initialized;

  void begin(float q, float r) {
    Q = q;
    R = r;
    P = 1.0;
    x = 0.0;
    initialized = false;
  }

  float update(float measurement) {
    if (isnan(measurement)) {
      return x;
    }
    if (!initialized) {
      x = measurement;
      initialized = true;
      return x;
    }
    P = P + Q;
    float K = P / (P + R);      
    x = x + K * (measurement - x);
    P = (1.0 - K) * P; // Corregido error tipográfico previo (1K) por (1.0 - K)
    return x;
  }
};

KalmanFilter1D kf_temp_in;
KalmanFilter1D kf_temp_out;
KalmanFilter1D kf_hum_in;
KalmanFilter1D kf_hum_out;

void ARDUINO_ISR_ATTR onTimer() {
  digitalWrite(LED, !digitalRead(LED));
  shouldSendRequest = true;
}

void setupPWM(int pin, int frequency, int resolution) {
  ledcAttach(pin, frequency, resolution);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED, OUTPUT);

  Serial.println("Iniciando sensores...");

  if (!aht10_in.begin(SDA_1, SCL_1, 100000)) {
    Serial.println("Error al inicializar AHT10 interior en SDA_1/SCL_1 (0x38)");
  } else {
    Serial.println("Sensor interior AHT10 inicializado correctamente (0x38).");
  }

  if (!aht10_out.begin(SDA_2, SCL_2, 100000)) {
    Serial.println("Error al inicializar AHT10 exterior en SDA_2/SCL_2 (0x39)");
  } else {
    Serial.println("Sensor exterior AHT10 inicializado correctamente (0x39).");
  }

  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();
  delay(10);

  setupPWM(ledPin, freq, resolution);
  setupPWM(HumPin, freq, resolution2);
  setupPWM(HeaterPin, freq, resolution2);
  setupPWM(heater_pin, freq2, resolution2);

  ledcWrite(HumPin, 20);
  ledcWrite(HeaterPin, dutyCycle);
  ledcWrite(ledPin, dutyCycle);
  ledcWrite(heater_pin, 0);

  timer = timerBegin(1000000); 
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, 10000000, true, 0); 

  // Inicializar filtros de Kalman
  kf_temp_in.begin(0.01, 0.0006);
  kf_temp_out.begin(0.01, 0.0002);
  kf_hum_in.begin(0.01, 0.0028);
  kf_hum_out.begin(0.01, 0.066);
}

void setVentanaPosition(int angle) {
  window = constrain(angle, 0, 90);
  
  if (!servoAttached) {
    VentServo.attach(SM);
    servoAttached = true;
  }
  
  VentServo.write(window);
  servoTimer = millis(); // Reiniciar el temporizador de inactividad
}

float readTemp(bool interior) {
  float raw = interior ? aht10_in.readTemperature() : aht10_out.readTemperature();
  return interior ? kf_temp_in.update(raw) : kf_temp_out.update(raw);
}

float readHum(bool interior) {
  float raw = interior ? aht10_in.readHumidity() : aht10_out.readHumidity();
  return interior ? kf_hum_in.update(raw) : kf_hum_out.update(raw);
}

void loop() {
  if (servoAttached && (millis() - servoTimer >= servoTimeout)) {
    VentServo.detach();
    servoAttached = false;
  }

  if (Serial.available()) {
    datain = Serial.readString();
    datain.trim(); 
    processCommand(datain);
  }

  WiFiClient client = server.available();
  if (client) {
    Serial.println("New Client.");
    String currentLine = "";
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        header += c;
        if (c == '\n') {
          if (currentLine.length() == 0) {
            sendWebPage(client);
            break;
          } else {
            currentLine = "";
          }
        } else if (c != '\r') {
          currentLine += c;
        }
      }
    }
    header = "";
    client.stop();
  }

  if (shouldSendRequest) {
    shouldSendRequest = false;
    sendDataToServer();
  }
}

void printRow(WiFiClient &client, const String &label, float in, float out, const String &unit) {
  client.printf(
    "<tr><td>%s</td><td><span class=\"sensor\">%.2f%s</span></td>"
    "<td><span class=\"sensor\">%.2f%s</span></td></tr>",
    label.c_str(), in, unit.c_str(), out, unit.c_str());
}

void printRowSingle(WiFiClient &client, const String &label, const String &value) {
  client.printf(
    "<tr><td>%s</td><td colspan=\"2\"><span class=\"sensor\">%s</span></td></tr>",
    label.c_str(), value.c_str());
}

void sendWebPage(WiFiClient &client) {
  client.println("HTTP/1.1 200 OK");
  client.println("Content-type:text/html");
  client.println("Connection: close");
  client.println();

  client.println("<!DOCTYPE html><html>");
  client.println("<head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
  client.println("<meta http-equiv=\"refresh\" content=\"10\">");
  client.println("<link rel=\"icon\" href=\"data:,\">");

  client.println("<style>body { text-align: center; font-family: \"Trebuchet MS\", Arial;}"
                 "table { border-collapse: collapse; width:50%; margin-left:auto; margin-right:auto; }"
                 "th { padding: 12px; background-color: #00af06; color: white; }"
                 "tr { border: 1px solid #ddd; padding: 12px; }"
                 "tr:hover { background-color: #bcbcbc; }"
                 "td { border: none; padding: 12px; }"
                 ".sensor { color:white; font-weight: bold; background-color: #bcbcbc; padding: 1px; }"
                 "</style></head><body><h1>Monitoreo y Control</h1>");

  client.println("<table><tr><th>PARAMETRO</th><th>Interior</th><th>Exterior</th></tr>");

  printRow(client, "Temp. Celsius", readTemp(true), readTemp(false), " *C");
  printRow(client, "Humedad Relativa", readHum(true), readHum(false), " %");

  printRowSingle(client, "Ventana", String(window) + "°");
  printRowSingle(client, "Calentador", String(heater));
  printRowSingle(client, "Humidificador", String(humifier));

  client.println("</table></body></html>");
  client.println();
}

void sendDataToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    http.setTimeout(15000);   
    http.begin(serverName);
    http.addHeader("Content-Type", "application/x-www-form-urlencoded");

    httpRequestData = "api_key=" + apiKeyValue
                    + "&value1=" + String(readTemp(true))
                    + "&value2=" + String(readTemp(false))
                    + "&value3=" + String(readHum(true))
                    + "&value4=" + String(readHum(false))
                    + "&value5=" + String(window)
                    + "&value6=" + String(heater)
                    + "&value7=" + String(humifier)
                    + "&value8=" + String(ref_temp)
                    + "&value9=" + String(ref_hum);

    int httpResponseCode = http.POST(httpRequestData);
    if (httpResponseCode > 0) {
      Serial.print("HTTP_OK:");
      Serial.println(httpResponseCode);
    } else {
      Serial.print("HTTP_ERROR:");
      Serial.println(httpResponseCode);
    }
    http.end();
  }
}

void processCommand(String command) {
  command.trim();

  if (command.length() < 2) {
    Serial.println("error:invalid_format");
    return;
  }

  char commandType = command.charAt(0);

  int valueStart = -1;
  for (int i = 1; i < command.length(); i++) {
    if (isDigit(command.charAt(i)) || command.charAt(i) == '.' || command.charAt(i) == '-') {
      valueStart = i;
      break;
    }
  }

  if (valueStart == -1) {
    Serial.println("error:no_value");
    return;
  }

  String valueStr = command.substring(valueStart);
  float value = valueStr.toFloat();

  bool success = true;
  String response = "cmd_";
  response += commandType;
  response += ":";

  switch (commandType) {
    case 'V': // Ventana (Usa la función segura de control de servo)
      setVentanaPosition(int(value));
      response += String(window);
      break;

    case 'C': // Calentador
      heater = constrain(int(value), 0, 255);
      ledcWrite(HeaterPin, heater);
      response += String(heater);
      break;

    case 'H': // Humidificador
      humifier = constrain(int(value), 0, 127);
      ledcWrite(HumPin, humifier);
      response += String(humifier);
      break;

    case 'A': // Temperatura referencia
      ref_temp = value;
      response += String(value, 2);
      break;

    case 'B': // Humedad referencia
      ref_hum = value;
      response += String(value, 2);
      break;

    case 'T': // Leer temperatura interior
      response += String(readTemp(true), 2);
      break;

    case 'Y': // Leer temperatura exterior
      response += String(readTemp(false), 2);
      break;

    case 'I': // Leer humedad interior
      response += String(readHum(true), 2);
      break;

    case 'J': // Leer humedad exterior
      response += String(readHum(false), 2);
      break;

    case 'S': // Leer todos los sensores
      response += "temp_in:";
      response += String(readTemp(true), 2);
      response += ",temp_out:";
      response += String(readTemp(false), 2);
      response += ",hum_in:";
      response += String(readHum(true), 2);
      response += ",hum_out:";
      response += String(readHum(false), 2);
      break;

    case 'R': // Leer valores de referencia
      response += "ref_temp:";
      response += String(ref_temp, 2);
      response += ",ref_hum:";
      response += String(ref_hum, 2);
      break;

    case 'D': // Leer estado de actuadores
      response += "window:";
      response += String(window);
      response += ",heater:";
      response += String(heater);
      response += ",humidifier:";
      response += String(humifier);
      break;

    default:
      success = false;
      response = "error:unknown_cmd";
      break;
  }

  Serial.println(response);
}
