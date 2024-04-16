#include <WiFi.h>
#include <ArduinoJson.h>

const char* ssid = "MEG";
const char* password = "098poilkjmnb";
WiFiServer server(80);
WiFiClient client;

const int m1a = 4;
const int m1b = 18;
const int m2a = 19;
const int m2b = 21;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());
  server.begin();
  pinMode(m1a, OUTPUT);
  pinMode(m1b, OUTPUT);
  pinMode(m2a, OUTPUT);
  pinMode(m2b, OUTPUT);
}

void loop() {
  client = server.available();
  if (client) {
    while (client.connected()) {
      if (client.available()) {
        String data = client.readStringUntil('\r');
//        Serial.println("Received from client: " + data[0]);
        // Process received data here

        // Parse JSON data
        StaticJsonDocument<200> doc; // Adjust the size as needed
        DeserializationError error = deserializeJson(doc, data);

        // Check for parsing errors
        if (error) {
          Serial.print("deserializeJson() failed: ");
          Serial.println(error.c_str());
          break;
        }

        // Accessing individual values from JSON object
        int right_speed_f = doc[0];
        int right_speed_r = doc[1];
        int left_speed_f = doc[2];
        int left_speed_r = doc[3];
        analogWrite(m1a, left_speed_f);
        analogWrite(m1b, left_speed_r);
        analogWrite(m2a, right_speed_f);
        analogWrite(m2b, right_speed_r);
//        Serial.println(right_speed_f);
        break;
      }
    }
    client.stop();
    Serial.println("Client disconnected");
  }
}
