#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <NewPing.h>
#include <Servo.h>

// Initialize the I2C LCD (address 0x27 is common, change if needed)
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Define ultrasonic sensor pins
#define TRIG_PIN 9
#define ECHO_PIN 10
#define MAX_DISTANCE 200

// Create a NewPing object for the ultrasonic sensor
NewPing sonar(TRIG_PIN, ECHO_PIN, MAX_DISTANCE);

// Servo motor object
Servo myServo;
#define SERVO_PIN 6

void clearScreen(void);
void setup() {
  // Initialize the LCD
  lcd.init();
  lcd.backlight(); // Turn on the backlight

  // Print initial message to the LCD
  lcd.setCursor(0, 0);
  lcd.print("Distance (cm):");

  // Initialize the servo motor
  myServo.attach(SERVO_PIN);
  myServo.write(0); // Start at 0 degrees (closed)

  // Start Serial for communication with Raspberry Pi
  Serial.begin(9600);
  Serial.println("Setup complete.");
}

void loop() {
  // Measure the distance using the ultrasonic sensor
  unsigned int distance = sonar.ping_cm();

  // Print the distance to the Serial Monitor (for debugging)
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // Clear the second row and print the distance on the LCD
  lcd.setCursor(0, 1);
  lcd.print("                 "); // Clear the second row
  lcd.setCursor(0, 1); // Reset cursor to the start of the second row

  if (distance > 0) {
    // Print the measured distance on the LCD
    lcd.setCursor(0, 0);
    lcd.print("Distance (cm):");
    lcd.setCursor(0, 1);
    lcd.print("                "); // Clear the second row
    lcd.setCursor(0, 1); // Reset cursor to the start of the second row
    lcd.print(distance);
    lcd.print(" cm");

    // If the distance is less than 10 cm, communicate with the Raspberry Pi
    if (distance < 10) {
      // Send a request to check parking availability
      Serial.println("CHECK_PARKING");

      // Wait for the response from Raspberry Pi
      long start_time = millis();
      bool response_received = false;

      while (millis() - start_time < 5000) {  // Wait for max 5 seconds
        if (Serial.available()) {
          String incoming_msg = Serial.readStringUntil('\n');
          incoming_msg.trim(); // Remove any newline or whitespace characters

          // If Raspberry Pi responds with "1", parking is full
          if (incoming_msg == "1") {
            clearScreen();
	    lcd.print("Parking Full");
            delay(2000);
            myServo.write(0); // Keep servo at 0 degrees (closed)
            response_received = true;
            break;
          } 
          // If Raspberry Pi responds with "0", parking is available
          else if (incoming_msg == "0") {
            clearScreen();
	    lcd.print("Parking Available");
            myServo.write(90); // Rotate servo to 90 degrees (open)
            while(sonar.ping_cm()<10);
            delay(4000); // Keep gate open for 4 seconds
            //while(sonar.ping_cm()<10);
            myServo.write(0); // Close the gate after the delay
            response_received = true;
            break;
          }
        }
      }

      // If no response is received within 5 seconds
      if (!response_received) {
	clearScreen();
        lcd.print("No Response");
        delay(2000); // Show "No Response" message for 2 seconds
        lcd.setCursor(0, 0); // Clear the message after delay
        lcd.print("                "); // Clear the first row
      }
    } else {
      // If the distance is more than 10 cm, reset the servo to 0 degrees (closed)
      myServo.write(0);
    }
  } else {
    // If the sensor doesn't detect any object
    lcd.print("Out of range");
    myServo.write(0); // Ensure the servo is at 0 degrees when out of range
  }

  // Delay before the next measurement
  delay(500);
}
void clearScreen(void){
   lcd.setCursor(0, 0);
   lcd.print("                ");
   lcd.setCursor(0, 1);
   lcd.print("                "); // Clear the second row
   lcd.setCursor(0, 0);
}
