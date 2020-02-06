// arduino serial reader

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()){
    int byte = Serial.read();
    Serial.println( byte );
  }
}
