#include <Arduino.h>

#include "firmware_configuration.h"
#include "watchdog_manager.h"
#include "boot_counter.h"
#include "time_manager.h"
#include "gnss_manager.h"
#include "sleep_manager.h"
#include "sd_card_manager.h"

static constexpr uint32_t SERIAL_TIMEOUT_MS = 5000;      ///< Max wait for serial connection

static constexpr bool ENABLE_BLINK_PWR_LED = false;          ///< Enable power LED blinking on startup
static constexpr bool ENABLE_BOOT_COUNTER = false;          ///< Enable boot counter functionality
static constexpr bool ENABLE_GNSS = false;                    ///< Enable GNSS module

void setup() {
  /////////////////////////////////////////////////////////////////////////////////
  // Initialize watchdog timer
  wdt.configure(WDT_1HZ, 32, 32);
  wdt.start();

  if (ENABLE_BLINK_PWR_LED){
    blink_pwr_led(3);
  }

  /////////////////////////////////////////////////////////////////////////////////
  // Initialize serial over USB
  SERIAL_USB->begin(BAUD_RATE_USB);
  while (!(*SERIAL_USB) && millis() < SERIAL_TIMEOUT_MS);

  /////////////////////////////////////////////////////////////////////////////////
  // Startup delay to allow time for serial monitor connection, uploading new firmware, etc.
  SERIAL_USB->println();
  SERIAL_USB->println(F("startup delay..."));
  wdt.restart();
  delay(500);
  wdt.restart();
  SERIAL_USB->println(F("... done"));
  SERIAL_USB->println();

  /////////////////////////////////////////////////////////////////////////////////
  // Print firmware configuration
  print_firmware_config();
  SERIAL_USB->println();

  /////////////////////////////////////////////////////////////////////////////////
  // Print boot count and offer to reset it
  // If the user presses 'y' within 5 seconds, reset the boot count
  // Otherwise, keep the current boot count
  if (ENABLE_BOOT_COUNTER){
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
  }

  if (ENABLE_BLINK_PWR_LED){
    blink_pwr_led(5);
  }

  /////////////////////////////////////////////////////////////////////////////////
  // Initialize time manager from GNSS
  // Keep trying to get a valid GNSS fix until successful
  // As long as we do not have a valid fix, sleep for a while between attempts
  board_time_manager.set_posix_timestamp(0);
  board_time_manager.print_status();
  SERIAL_USB->println();
  if (ENABLE_GNSS){
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
  }

  if (ENABLE_BLINK_PWR_LED){
    blink_pwr_led(7);
  }
  
  /////////////////////////////////////////////////////////////////////////////////
  // start I2C port

  // start and set up GNSS itself

  // set up interrupt on PPS

  // start and set up ISM330DHCX

  /////////////////////////////////////////////////////////////////////////////////
  // log forever

  // Start doing the logging to the SD card
  sd_card_manager.start();

  constexpr uint32_t PREALLOCATE_SIZE_BYTES = 100 * 1024 * 1024; // Preallocate 100 MB

  if (sd_card_manager.preallocate_and_open_file(PREALLOCATE_SIZE_BYTES)) {
    SERIAL_USB->println(F("Log file opened and preallocated successfully."));
    sd_card_manager.write_buffer((const uint8_t *)"Log start\n", 10);
    delay(100);
    sd_card_manager.close_and_sync_file();
  } else {
    SERIAL_USB->println(F("ERROR: Failed to open and preallocate log file on SD card."));
  }

  // TODO:
  // GNSS: 10Hz, lots of data logged
  // GNSS: PPS
  // IMU: max quality
  // every 15 minutes: new file; filename: boot_fileindex_YYYYMMDD_HHMMSS.dat
  // with interrupt


  /////////////////////////////////////////////////////////////////////////////////
  // if we reach here, we have an issue:
  // stop SD card manager and let watchdog reset the board
  sd_card_manager.close_and_sync_file();
  delay(1000);
  sd_card_manager.stop();

  while (true)
  {
    // reboot
  }

}

void loop() {
  // we should never get here
  // if we get here, the watchdog will eventually restart the board
}
