# Monitoreo Control Invernadero v1

Sistema web de monitoreo en tiempo real para un invernadero controlado por ESP32, compuesto por un endpoint de ingesta (`POST`), un endpoint de consulta (`JSON`) y un panel visual con **Highcharts**.

## Arquitectura y archivos

- **`Post-data.php`**: Recibe lecturas por `POST` (protegidas por `api_key`), limpia las entradas con `test_input()` y guarda `value1` a `value9` en la tabla `MPC_data` mediante consulta preparada (`mysqli::prepare`).
- **`datos.php`**: Valida `time_range` contra una lista blanca (`1 HOUR`, `1 DAY`, `7 DAY`; por defecto `1 DAY`), consulta `MPC_data`, ajusta la zona horaria (`-4` horas) y devuelve los datos en JSON con marcas de tiempo en milisegundos.
- **`index.html`**: Interfaz web que carga los datos al abrirse y actualiza cada 5 segundos tres gráficas de Highcharts:
  1. **Temperatura** (`value1`: interna, `value2`: externa, `value8`: referencia).
  2. **Humedad %** (`value3`: interna, `value4`: externa, `value9`: referencia).
  3. **Actuadores** (`value5`: ventana, `value6`: calentador, `value7`: humidificador).

## Mapeo de variables (`MPC_data`)

| Campo DB | Significado en el dashboard | Unidad / Rango típico |
| :--- | :--- | :--- |
| `value1` | Temperatura interna | `°C` |
| `value2` | Temperatura externa | `°C` |
| `value3` | Humedad interna | `%` |
| `value4` | Humedad externa | `%` |
| `value5` | Actuador: Ventana | `0 - 255` |
| `value6` | Actuador: Calentador | `0 - 255` |
| `value7` | Actuador: Humidificador | `0 - 255` |
| `value8` | Temperatura de referencia | `°C` |
| `value9` | Humedad de referencia | `%` |
| `reading_time` | Marca de tiempo del registro | `DATETIME` |

## Configuración y despliegue

1. **Base de datos**: Crea la base de datos y la tabla `MPC_data` con las columnas `value1`..`value9` y `reading_time` (por ejemplo, `TIMESTAMP DEFAULT CURRENT_TIMESTAMP`).
2. **Credenciales**: Configura host, base de datos, usuario, contraseña y `api_key_value` en `Post-data.php` y `datos.php` (idealmente mediante variables de entorno o un archivo de configuración fuera de la carpeta pública).
3. **Servidor web**: Sube `Post-data.php`, `datos.php` e `index.html` a un servidor con soporte PHP y extensión `mysqli`.

## Ejemplo de envío (`POST`)

```bash
curl -X POST [https://tu-dominio.com/Post-data.php](https://tu-dominio.com/Post-data.php) \
  -d "api_key=TU_API_KEY" \
  -d "value1=22.5&value2=18.0&value3=65&value4=55&value5=120&value6=0&value7=80&value8=23.0&value9=65"
