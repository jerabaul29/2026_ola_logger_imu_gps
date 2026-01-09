#include <Arduino.h>

#include "firmware_configuration.h"
#include "watchdog_manager.h"
#include "boot_counter.h"
#include "time_manager.h"
#include "gnss_manager.h"
#include "sleep_manager.h"

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
  SERIAL_USB->println(F("Press y to reset boot count... "));
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

  // Initialize time manager from GNSS
  // Keep trying to get a valid GNSS fix until successful
  // As long as we do not have a valid fix, sleep for a while between attempts
  board_time_manager.set_posix_timestamp(0);
  board_time_manager.print_status();
  SERIAL_USB->println();
  bool got_valid_fix = false;
  while (!got_valid_fix) {
    SERIAL_USB->println(F("Attempting to get initial GNSS fix..."));
    wdt.restart();
    got_valid_fix = gnss_manager.get_a_fix(timeout_initial_fix_gnss_seconds, false, true, false);
    if (!got_valid_fix) {
      SERIAL_USB->println(F("Failed to get GNSS fix. Sleep and retry..."));
      turn_gnss_off();
      wdt.restart();
      sleep_for_seconds(sleep_no_initial_gnss_fix_seconds);
      blink_stat_led(3);
    }
    else {
      SERIAL_USB->println(F("Successfully obtained GNSS fix."));
      SERIAL_USB->println(F("Get a few fixes to make sure the quality is good before setting clock..."));
      for (int i=0; i<10; i++) {
        wdt.restart();
        gnss_manager.get_a_fix(10, false, false, false);
      }
      gnss_manager.get_a_fix(10, true, false, false);
    }
  }
  board_time_manager.print_status();
  SERIAL_USB->println();

  blink_pwr_led(7);
  
  // Start doing the logging to the SD card

  // TODO:
  // GNSS: 10Hz, lots of data logged
  // GNSS: PPS
  // IMU: max quality
  // every 15 minutes: new file; filename: boot_fileindex_YYYYMMDD_HHMMSS.dat
  // with interrupt

  while (true){
    wdt.restart();
    delay(1000);
  }

}

void loop() {
  // we should never get here
  // if we get here, the watchdog will eventually restart the board
}
