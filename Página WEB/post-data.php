<?php
$servername = "****.runhosting.com";
$dbname = "****_esp32data";
$username = "*****_esp32data";
$password = "*****";
$api_key_value = "*****";

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $input_api_key = test_input($_POST["api_key"] ?? "");

    if ($input_api_key === $api_key_value) {
        $values = [];
        for ($i = 1; $i <= 9; $i++) {
            $values[] = test_input($_POST["value$i"] ?? "");
        }

        $conn = new mysqli($servername, $username, $password, $dbname);
        if ($conn->connect_error) {
            die("Connection failed: " . $conn->connect_error);
        }

        $stmt = $conn->prepare(
            "INSERT INTO MPC_data (value1, value2, value3, value4, value5, value6, value7, value8, value9)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        );
        $stmt->bind_param("sssssssss", ...$values);

        if ($stmt->execute()) {
            echo "New record created successfully";
        } else {
            echo "Error: " . $stmt->error;
        }

        $stmt->close();
        $conn->close();
    } else {
        echo "Wrong API Key provided.";
    }
} else {
    echo "No data posted with HTTP POST. Marcelo Mendoza";
}

function test_input($data) {
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}