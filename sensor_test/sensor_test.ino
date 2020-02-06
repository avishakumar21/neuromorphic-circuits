
// code to read the voltage values outputted by each sensor 

#include "Arduino.h"

int PinOut = 13;

// Sample time
unsigned long timeS;

// String output:
String Output15;

// Set counters to zero
int value = 0;

// PWM output pins. This disables timer1 and timer2
int PinPWM1 = 12;
int PinPWM2 = 11;
int PinPWM3 = 10;
int PinPWM4 = 9;

// Initialize input readings
int ch1 = 0;
int ch2 = 0;
int ch3 = 0;
int ch4 = 0;
int ch5 = 0;
int ch6 = 0;
int ch7 = 0;
int ch8 = 0;
int ch9 = 0;
int ch10 = 0;
int ch11 = 0;
int ch12 = 0;
int ch13 = 0;
int ch14 = 0;
int ch15 = 0;
int ch16 = 0;

int ACh1 = 8;
int ACh2 = 15;
int ACh3 = 4;
int ACh4 = 3;
int ACh5 = 2;
int ACh6 = 9;
int ACh7 = 1;
int ACh8 = 0;
int ACh9 = 5;
int ACh10 = 14;
int ACh11 = 7;
int ACh12 = 6;
int ACh13 = 10;
int ACh14 = 11;
int ACh15 = 13;
int ACh16 = 12;

// Chip Select
int CS1 = 28;
int CS2 = 44;
int CS3 = 49;
int CS4 = 51;
int CS5 = 32;
int CS6 = 42;
int CS7 = 48;
int CS8 = 50;
int CS9 = 34;
int CSi0 = 40;
int CSi1 = 47;
int CSi2 = 52;
int CSi3 = 26;
int CSi4 = 38;
int CSi5 = 46;
int CSi6 = 53;

// U/D' 
int pinUD = 30;


void setup(){
  // initialize serial communication.
  Serial.begin(115200); // set the baud rate
  Serial.println("Ready"); // print "Ready" once
  

  printf("printing voltage values of 16 mox sensors\n");
  
  
  analogReference(EXTERNAL); //Set voltage reference to external. 
  
  // Set the duty cycle for the PWM. Set higher PWM for higher voltage -> higher temperature. PWM =( 0,255)
  pinMode(PinPWM1, OUTPUT);
  analogWrite(PinPWM1, 160); //160: 2.57V. Vheater: 4.43V
  pinMode(PinPWM2, OUTPUT);
  analogWrite(PinPWM2, 192); //192: 1.71V. Vheater: 5.29V
  pinMode(PinPWM3, OUTPUT);
  analogWrite(PinPWM3, 205); //205: 1.5V. Vheater: 5.65V
  pinMode(PinPWM4, OUTPUT);
  analogWrite(PinPWM4, 225); //225: 0.8V. Vheater: 6.2V


  // initialize digital pin PinOut as an output.
  pinMode(PinOut, OUTPUT);
  digitalWrite(PinOut, LOW);

  
  // initialize timer3
  noInterrupts();           // disable all interrupts
  TCCR3A = 0;
  TCCR3B = 0;
  TCNT3  = 0;

  // OCR3A = 31250;            // compare match register 16MHz/256/2Hz
  OCR3A = 6250;            // compare match register 16MHz/256/10Hz
  TCCR3B |= (1 << WGM12);   // CTC mode
  TCCR3B |= (1 << CS12);    // 256 prescaler
  TIMSK3 |= (1 << OCIE1A);  // enable timer compare interrupt
  interrupts();             // enable all interrupts


  // initialize digital pin ChipSelect as outputs.
  pinMode(CS1, OUTPUT);
  digitalWrite(CS1, HIGH);
  pinMode(CS2, OUTPUT);
  digitalWrite(CS2, HIGH);
  pinMode(CS3, OUTPUT);
  digitalWrite(CS3, HIGH);
  pinMode(CS4, OUTPUT);
  digitalWrite(CS4, HIGH);
  pinMode(CS5, OUTPUT);
  digitalWrite(CS5, HIGH);
  pinMode(CS6, OUTPUT);
  digitalWrite(CS6, HIGH);
  pinMode(CS7, OUTPUT);
  digitalWrite(CS7, HIGH);
  pinMode(CS8, OUTPUT);
  digitalWrite(CS8, HIGH);
  pinMode(CS9, OUTPUT);
  digitalWrite(CS9, HIGH);
  pinMode(CSi0, OUTPUT);
  digitalWrite(CSi0, HIGH);
  pinMode(CSi1, OUTPUT);
  digitalWrite(CSi1, HIGH);
  pinMode(CSi2, OUTPUT);
  digitalWrite(CSi2, HIGH);
  pinMode(CSi3, OUTPUT);
  digitalWrite(CSi3, HIGH);
  pinMode(CSi4, OUTPUT);
  digitalWrite(CSi4, HIGH);
  pinMode(CSi5, OUTPUT);
  digitalWrite(CSi5, HIGH);
  pinMode(CSi6, OUTPUT);
  digitalWrite(CSi6, HIGH);

  
  // initialize digital pin UD as output.
  pinMode(pinUD, OUTPUT);
  digitalWrite(pinUD, HIGH);


  // configure load resistor to about 10K
  Init_R(CS1);
  Init_R(CS2);
  Init_R(CS3);
  Init_R(CS4);
  Init_R(CS5);
  Init_R(CS6);
  Init_R(CS7);
  Init_R(CS8);
  Init_R(CS9);
  Init_R(CSi0);
  Init_R(CSi1);
  Init_R(CSi2);
  Init_R(CSi3);
  Init_R(CSi4);
  Init_R(CSi5);  
  Init_R(CSi6);
}




void Init_R(int cha)
{

  // bring variable resistor to lowest resistance (75 Ohm)
  digitalWrite(pinUD, LOW);
  delay(1);
  digitalWrite(cha, LOW);
  delay(1);
  
  for (int i=0; i<50; i++)
  {
  //Serial.print(i);

  digitalWrite(pinUD, HIGH);
  delay(2);
  
  digitalWrite(pinUD, LOW);
  delay(2);
    
  }
  digitalWrite(cha, HIGH);
  delay(2);

  // bring resistor to about 6.7KOhm (there is a 3.3K resistor in series that adds to the load resistor)
  digitalWrite(pinUD, HIGH);
  delay(1);
  digitalWrite(cha, LOW);
  delay(1);

  for (int i=0; i<8; i++)
  {

    digitalWrite(pinUD, LOW);
    delay(2);
    
    digitalWrite(pinUD, HIGH);
    delay(2);
    
  }
  digitalWrite(cha, HIGH);
  delay(2);


}

void loop() {

  if (millis()%1000==0){ // print every one second 
    
      int lectura;
  
      Serial.print("Ch1:");
      lectura = analogRead(ACh1);
      Serial.print(lectura);
      
      Serial.print("; Ch2:");
      lectura = analogRead(ACh2);
      Serial.print(lectura);  
    
      Serial.print("; Ch3:");
      lectura = analogRead(ACh3);
      Serial.print(lectura);
      
      Serial.print("; Ch4:");
      lectura = analogRead(ACh4);
      Serial.print(lectura);  
    
      Serial.print("; Ch5:");
      lectura = analogRead(ACh5);
      Serial.print(lectura); 
      
      Serial.print("; Ch6:");
      lectura = analogRead(ACh6);
      Serial.print(lectura); 
    
      Serial.print("; Ch7:");
      lectura = analogRead(ACh7);
      Serial.print(lectura); 
      
      Serial.print("; Ch8:");
      lectura = analogRead(ACh8);
      Serial.print(lectura);
    
      Serial.print(" Ch9:");
      lectura = analogRead(ACh9);
      Serial.print(lectura);
      
      Serial.print("; Ch10:");
      lectura = analogRead(ACh10);
      Serial.print(lectura);  
    
      Serial.print("; Ch11:");
      lectura = analogRead(ACh11);
      Serial.print(lectura);
      
      Serial.print("; Ch12:");
      lectura = analogRead(ACh12);
      Serial.print(lectura);  
    
      Serial.print("; Ch13:");
      lectura = analogRead(ACh13);
      Serial.print(lectura); 
      
      Serial.print("; Ch14:");
      lectura = analogRead(ACh14);
      Serial.print(lectura); 
    
      Serial.print("; Ch15:");
      lectura = analogRead(ACh15);
      Serial.print(lectura); 
      
      Serial.print("; Ch16:");
      lectura = analogRead(ACh16);
      Serial.println(lectura);    
  }
}


ISR(TIMER3_COMPA_vect)          // timer compare interrupt service routine
{

  analogRead(ACh1);
  ch1 = analogRead(ACh1);
  
  analogRead(ACh2);
  ch2 = analogRead(ACh2);
  
  analogRead(ACh3);
  ch3 = analogRead(ACh3);
  
  analogRead(ACh4);
  ch4 = analogRead(ACh4);
  
  analogRead(ACh5);
  ch5 = analogRead(ACh5);
  
  analogRead(ACh6);
  ch6 = analogRead(ACh6);
  
  analogRead(ACh7);
  ch7 = analogRead(ACh7);
  
  analogRead(ACh8);
  ch8 = analogRead(ACh8);
  
  analogRead(ACh9);
  ch9 = analogRead(ACh9);
  
  analogRead(ACh10);
  ch10 = analogRead(ACh10);
  
  analogRead(ACh11);
  ch11 = analogRead(ACh11);
  
  analogRead(ACh12);
  ch12 = analogRead(ACh12);
  
  analogRead(ACh13);
  ch13 = analogRead(ACh13);
  
  analogRead(ACh14);
  ch14 = analogRead(ACh14);
  
  analogRead(ACh15);
  ch15 = analogRead(ACh15);
  
  analogRead(ACh16);
  ch16 = analogRead(ACh16);

  timeS = millis();

  
}



