#include <Arduino.h>

#include "firmware_configuration.h"
#include "watchdog_manager.h"
#include "boot_counter.h"

static constexpr uint32_t SERIAL_TIMEOUT_MS = 5000;      ///< Max wait for serial connection

void setup() {
  // Initialize watchdog timer
  wdt.configure(WDT_1HZ, 32, 32);
  wdt.start();

  blink_pwr_led(3);

  // Initialize serial over USB
  SERIAL_USB->begin(BAUD_RATE_USB);
  while (!(*SERIAL_USB) && millis() < SERIAL_TIMEOUT_MS);

  // Startup delay to allow time for serial monitor connection, uploading new firmware, etc.
  SERIAL_USB->println();
  SERIAL_USB->println(F("startup delay..."));
  wdt.restart();
  delay(1000);
  wdt.restart();
  SERIAL_USB->println(F("... done"));
  SERIAL_USB->println();

  // Print firmware configuration
  print_firmware_config();
  SERIAL_USB->println();

  // Print boot count and offer to reset it
  // If the user presses 'y' within 5 seconds, reset the boot count
  // Otherwise, keep the current boot count
  SERIAL_USB->print(F("Boot count: "));
  SERIAL_USB->println(boot_counter_instance.get_boot_number());
  SERIAL_USB->print(F("Press y to reset boot count... "));
  wdt.restart();
  unsigned long startTime = millis();
  bool resetRequested = false;
  while (millis() - startTime < 5000) {
    wdt.restart();
    if (SERIAL_USB->available()) {
      char c = SERIAL_USB->read();
      if (c == 'y' || c == 'Y') {
        resetRequested = true;
        break;
      }
    }
  }
  if (resetRequested) {
    boot_counter_instance.set_boot_number(0);
    SERIAL_USB->println(F("Boot count reset."));
  } else {
    SERIAL_USB->println(F("No reset requested."));
    boot_counter_instance.increment_boot_number();
    delay(100);
    wdt.restart();
  }
  SERIAL_USB->println();

  blink_pwr_led(5);
}

void loop() {
  wdt.restart();
}
