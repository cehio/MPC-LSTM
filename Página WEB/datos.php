<?php
$servername = "***";
$dbname = "****_esp32data";
$username = "****_esp32data";
$password = "****";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$allowed_ranges = [
    "1 HOUR" => "1 HOUR",
    "1 DAY"  => "1 DAY",
    "7 DAY"  => "7 DAY",
];
$requested_range = $_GET["time_range"] ?? "1 DAY";
$time_range = $allowed_ranges[$requested_range] ?? "1 DAY";

$sql = "SELECT value1, value2, value3, value4, value5, value6, value7, value8, value9, reading_time
        FROM MPC_data
        WHERE reading_time > NOW() - INTERVAL $time_range
        ORDER BY reading_time DESC";

$result = $conn->query($sql);

$response = [];
for ($i = 1; $i <= 9; $i++) {
    $response["value$i"] = [];
}

if ($result && $result->num_rows > 0) {
    while ($row = $result->fetch_assoc()) {
        $timestamp = (strtotime($row["reading_time"]) - 4 * 3600) * 1000;
        for ($i = 1; $i <= 9; $i++) {
            $response["value$i"][] = [$timestamp, $row["value$i"]];
        }
    }
    $result->free();
}

header("Content-Type: application/json");
echo json_encode($response, JSON_NUMERIC_CHECK);

$conn->close();