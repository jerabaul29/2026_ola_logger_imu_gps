#include <Arduino.h>

#include "firmware_configuration.h"
#include "watchdog_manager.h"
#include "boot_counter.h"
#include "time_manager.h"
#include "gnss_manager.h"
#include "sleep_manager.h"
#include "sd_card_manager.h"

#include "Wire.h"
#include <SparkFun_u-blox_GNSS_v3.h>

#include <ISM330DHCXSensor.h>

#include "Embedded_Template_Library.h"
#include "etl/deque.h"

// Timer configuration
static constexpr int TIMER_NUM = 2;
static constexpr uint32_t TIMER_FREQ_HZ = 1000;

int32_t acc_value[3];
int32_t gyr_value[3];

bool acc_available = false;
bool gyr_available = false;

SFE_UBLOX_GNSS log_GNSS;
static constexpr uint32_t GNSS_FREQUENCY_HZ = 10;

static constexpr uint32_t SERIAL_TIMEOUT_MS = 5000;      ///< Max wait for serial connection

static constexpr bool ENABLE_BLINK_PWR_LED = true;          ///< Enable power LED blinking on startup
static constexpr bool ENABLE_BOOT_COUNTER = true;          ///< Enable boot counter functionality
static constexpr bool ENABLE_GNSS_START = true;                    ///< Enable GNSS module

TwoWire * I2C_QWIIC = &Wire1;

ISM330DHCXSensor AccGyr(I2C_QWIIC, 0x6A);

static constexpr uint32_t seconds_in_15_minutes = 15 * 60;

constexpr uint32_t PREALLOCATE_LOGFILE_SIZE_BYTES = 25 * 1024 * 1024; // Preallocate a file large enough for logging

static constexpr char str_start_logging[] = "Log start\n\n";
static constexpr char str_stop_logging[] = "\n\nLog stop\n";

static constexpr size_t SIZE_DEQUES {2048};

struct PPS_fix {
  unsigned long millis_reading;
};

struct GNSS_reading {
  unsigned long millis_reading;
  int32_t latitude;
  int32_t longitude;
  uint32_t posix_timestamp;
  uint32_t microseconds;
  int32_t NED_vel_north;
  int32_t NED_vel_east;
  int32_t NED_vel_down;
  uint8_t fix_type;
};

struct ACC_reading{
  unsigned long millis_reading;
  int32_t acc_x;
  int32_t acc_y;
  int32_t acc_z;
};

// TODO: do GYR

char entry_kind[4];
bool should_log_data = false;

PPS_fix common_pps_fix;
GNSS_reading common_gnss_reading;
ACC_reading common_acc_reading;

etl::deque<PPS_fix, SIZE_DEQUES> deque_PPS_fixes;
etl::deque<GNSS_reading, SIZE_DEQUES> deque_GNSS_readings;
etl::deque<ACC_reading, SIZE_DEQUES> deque_ACC_readings;

volatile uint32_t ctimer_isr_count {0};

// ISR handler for CTIMER interrupts
// we use teh CTIMER to generate periodic interrupts for the data logging tasks
extern "C" void am_ctimer_isr(void)
{
  // Get interrupt status and clear
  uint32_t ui32Status = am_hal_ctimer_int_status_get(true);
  am_hal_ctimer_int_clear(ui32Status);
  
  // Check if it's timer 2A interrupt (reload/overflow)
  if (ui32Status & AM_HAL_CTIMER_INT_TIMERA2)
  {
    // do our work here

    // read IMU data and store in deque as many as fifo entries
    uint16_t num_samples_available = 0;
    AccGyr.FIFO_Get_Num_Samples(&num_samples_available);
    if (num_samples_available > 1){
      for (uint16_t i=0; i<num_samples_available; i++){
        uint8_t tag;
        // Check the FIFO tag
        AccGyr.FIFO_Get_Tag(&tag);
        switch (tag) {
          // If we have a gyro tag, read the gyro data
          case ISM330DHCX_GYRO_NC_TAG: {
              AccGyr.FIFO_GYRO_Get_Axes(gyr_value);
              gyr_available = true;
              break;
            }
          // If we have an acc tag, read the acc data
          case ISM330DHCX_XL_NC_TAG: {
              AccGyr.FIFO_ACC_Get_Axes(acc_value);
              acc_available = true;
              break;
            }
          // We can discard other tags
          default: {
              break;
            }
        }

        if (acc_available){
          common_acc_reading.millis_reading = millis();
          common_acc_reading.acc_x = acc_value[0];
          common_acc_reading.acc_y = acc_value[1];
          common_acc_reading.acc_z = acc_value[2];

          deque_ACC_readings.push_back(common_acc_reading);

          acc_available = false;
        }
      }
    }

    // if time to read GNSS data, do it and store in deque
    if (ctimer_isr_count % (TIMER_FREQ_HZ / GNSS_FREQUENCY_HZ) == 0){
      // check if we have a new GNSS reading; if yes, push fix to deque
      if (log_GNSS.getPVT()){
        common_gnss_reading.millis_reading = millis();
        common_gnss_reading.latitude = log_GNSS.getLatitude();
        common_gnss_reading.longitude = log_GNSS.getLongitude();
        common_gnss_reading.posix_timestamp = log_GNSS.getUnixEpoch(common_gnss_reading.microseconds);
        common_gnss_reading.NED_vel_north = log_GNSS.getNedNorthVel();
        common_gnss_reading.NED_vel_east = log_GNSS.getNedEastVel();
        common_gnss_reading.NED_vel_down = log_GNSS.getNedDownVel();
        common_gnss_reading.fix_type = log_GNSS.getFixType();

        deque_GNSS_readings.push_back(common_gnss_reading);
    }
  }

  ctimer_isr_count += 1;
  }
}

// TODO: PPS ISR handler

void setup() {
  /////////////////////////////////////////////////////////////////////////////////
  // Initialize watchdog timer
  wdt.configure(WDT_1HZ, 32, 32);
  wdt.start();

  if (ENABLE_BLINK_PWR_LED){
    blink_pwr_led(3);
  }

  pinMode(PIN_STAT_LED, OUTPUT);

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

  boot_counter_instance.increment_boot_number();
  delay(100);
  wdt.restart();

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
  if (ENABLE_GNSS_START){
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

  int start_attempt {1};

  while (start_attempt<=5){
    SERIAL_USB->print(F("Setup attempt #: "));
    SERIAL_USB->println(start_attempt);
    start_attempt += 1;

    wdt.restart();
    delay(10);
    
    ////////////////////////////////////////////////////
    // start I2C port

    SERIAL_USB->println(F("Starting I2C QWIIC..."));
    pinMode(PIN_QWIIC_PWR, OUTPUT);
    digitalWrite(PIN_QWIIC_PWR, HIGH); 
    delay(100);
    wdt.restart();

    I2C_QWIIC->begin();
    delay(100);
    wdt.restart();
    I2C_QWIIC->setClock(400000);
    delay(100);
    wdt.restart();
    SERIAL_USB->println(F("I2C QWIIC started"));

    ////////////////////////////////////////////////////
    // start and set up GNSS itself

    if (!log_GNSS.begin(*I2C_QWIIC)){
        SERIAL_USB->println(F("problem starting GNSS"));
        
        I2C_QWIIC->end();
        delay(500);
        continue;
    }
    SERIAL_USB->println(F("success starting GNSS"));

    log_GNSS.setI2COutput(COM_TYPE_UBX);
    delay(100);
    wdt.restart();
    SERIAL_USB->println(F("GNSS set to UBX output"));
    delay(100);

    // if (!gnss.setDynamicModel(DYN_MODEL_PORTABLE)){
    //   SERIAL_USB->println(F("GNSS could not set dynamic model"));
    //   continue;
    // }
    // SERIAL_USB->println(F("GNSS dynamic model set to PORTABLE"));

    log_GNSS.setAutoPVT(true);
    log_GNSS.setNavigationFrequency(GNSS_FREQUENCY_HZ);
    delay(100);
    wdt.restart();
    uint8_t rate = log_GNSS.getNavigationFrequency();
    SERIAL_USB->print("Current update rate: ");
    SERIAL_USB->println(rate);
 
    // wait until we get a fix
    bool fix_obtained {false};
    static constexpr unsigned long GNSS_FIX_WAIT_TIMEOUT_MS = 1000 * 60 * 2;
    unsigned long start_wait_ms = millis();
    SERIAL_USB->println(F("Waiting for GNSS fix..."));
    while (millis() - start_wait_ms < GNSS_FIX_WAIT_TIMEOUT_MS){
      if (log_GNSS.getFixType() >= 3){
        fix_obtained = true;
        SERIAL_USB->println(F("GNSS fix acquired."));
        break;
      }
      delay(500);
      wdt.restart();
      SERIAL_USB->print(F("."));
    }

    if (!fix_obtained){
      SERIAL_USB->println();
      SERIAL_USB->println(F("Failed to obtain GNSS fix in time."));
      continue;
    }

    SERIAL_USB->println(F("GNSS setup complete."));
    wdt.restart();

    ////////////////////////////////////////////////////
    // start and set up ISM330DHCX

    SERIAL_USB->println(F("Starting ISM330DHCX..."));

    if (AccGyr.begin() != ISM330DHCX_OK){
      SERIAL_USB->println(F("problem starting ISM330DHCX"));
      
      log_GNSS.end();
      I2C_QWIIC->end();
      delay(500);
      continue;
    }
    SERIAL_USB->println(F("success starting ISM330DHCX"));
    delay(100);
    wdt.restart();

    AccGyr.ACC_Enable();
    delay(10);
    AccGyr.GYRO_Enable();
    delay(10);
    wdt.restart();

    AccGyr.ACC_SetOutputDataRate(833.0f);
    delay(10);
    AccGyr.GYRO_SetOutputDataRate(833.0f);
    delay(10);
    wdt.restart();

    AccGyr.ACC_SetFullScale(ISM330DHCX_2g);
    delay(10);
    AccGyr.GYRO_SetFullScale(ISM330DHCX_125dps);
    delay(10);
    wdt.restart();

    AccGyr.FIFO_ACC_Set_BDR(833.0f);
    delay(10);
    AccGyr.FIFO_GYRO_Set_BDR(833.0f);
    delay(10);
    wdt.restart();

    AccGyr.FIFO_Set_Mode(ISM330DHCX_STREAM_MODE);
    delay(10);
    wdt.restart();

    SERIAL_USB->println(F("ISM330DHCX setup complete."));

    ////////////////////////////////////////////////////
    SERIAL_USB->println(F("All set up, ready to log: start isr timer..."));

    // Power up the clock
    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);
    
    // Enable global interrupts
    am_hal_interrupt_master_enable();
    
    // Stop timer
    am_hal_ctimer_stop(TIMER_NUM, AM_HAL_CTIMER_TIMERA);
    
    // Clear timer
    am_hal_ctimer_clear(TIMER_NUM, AM_HAL_CTIMER_TIMERA);
    
    // Configure timer in REPEAT mode with 3MHz clock
    am_hal_ctimer_config_single(TIMER_NUM, AM_HAL_CTIMER_TIMERA,
                                (AM_HAL_CTIMER_FN_REPEAT | 
                                  AM_HAL_CTIMER_HFRC_3MHZ |
                                  AM_HAL_CTIMER_INT_ENABLE));
    
    // Set the period for the timer
    uint32_t period = 3000000 / TIMER_FREQ_HZ;
    am_hal_ctimer_period_set(TIMER_NUM, AM_HAL_CTIMER_TIMERA, period, 0);
    
    // Clear any pending interrupts
    am_hal_ctimer_int_clear(AM_HAL_CTIMER_INT_TIMERA2);
    
    // Enable the timer interrupt in main CTIMER register
    am_hal_ctimer_int_enable(AM_HAL_CTIMER_INT_TIMERA2);
    
    // Enable interrupt at NVIC level
    NVIC_EnableIRQ(CTIMER_IRQn);
    
    // Start the timer
    am_hal_ctimer_start(TIMER_NUM, AM_HAL_CTIMER_TIMERA);
    Serial.println(F("Timer started!"));

    break;
  }

  /////////////////////////////////////////////////////////////////////////////////
  // log forever

  SERIAL_USB->println(F("Clear buffers"));
  deque_PPS_fixes.clear();
  deque_GNSS_readings.clear();
  deque_ACC_readings.clear();

  // Start doing the logging to the SD card
  SERIAL_USB->println(F("Starting SD card manager..."));
  sd_card_manager.start();
  SERIAL_USB->println(F("SD card manager started."));

  uint32_t posix_timestamp;
  uint32_t posix_timestamp_next_file;

  while (true){
    // create a new file
    SERIAL_USB->println(F("Preparing to start new log file..."));

    if (sd_card_manager.preallocate_and_open_file(PREALLOCATE_LOGFILE_SIZE_BYTES)) {
      SERIAL_USB->println(F("Log file opened and preallocated successfully."));
    } else {
      SERIAL_USB->println(F("ERROR: Failed to open and preallocate log file on SD card."));
      break;
    }
    wdt.restart();

    // we do a new file every time UTC times hits 0 minutes modulo 15 minutes
    posix_timestamp = board_time_manager.get_posix_timestamp();
    SERIAL_USB->print(F("Current posix timestamp: "));
    SERIAL_USB->println(posix_timestamp);
    posix_timestamp_next_file = posix_timestamp - (posix_timestamp % seconds_in_15_minutes) + seconds_in_15_minutes;
    SERIAL_USB->print(F("Next log file posix timestamp: "));
    SERIAL_USB->println(posix_timestamp_next_file);
    wdt.restart();
    SERIAL_USB->println(F("Logging..."));

    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(str_start_logging), sizeof(str_start_logging)-1);
    wdt.restart();

    while (board_time_manager.get_posix_timestamp() < posix_timestamp_next_file){
      wdt.restart();

      // log
      // the logging from sensors to dequeues buffers is taken care of by the ISR driven routines

      // if there are data on the dequeue buffers, write them to SD card
      // for each of the deques:
      //   - turn off interrupts as the deques are shared with ISRs
      //   - pop from the deques into the local buffer to make ready to write
      //   - turn on interrupts
      //   - write the local buffer to SD card

      // with the GNSS PPS deque
      am_hal_interrupt_master_disable();

      if (deque_PPS_fixes.size() > 0){
        should_log_data = true;
        common_pps_fix = deque_PPS_fixes.front();
        deque_PPS_fixes.pop_front();
      }

      am_hal_interrupt_master_enable();

      if (should_log_data){
        entry_kind[0] = '\n';
        entry_kind[1] = 'P';
        entry_kind[2] = 'P';
        entry_kind[3] = 'S';
        sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(entry_kind), sizeof(entry_kind));
        sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(&common_pps_fix), sizeof(common_pps_fix));
        should_log_data = false;
      }

      // with the GNSS fixes deque
      am_hal_interrupt_master_disable();

      if (deque_GNSS_readings.size() > 0){
        should_log_data = true;
        common_gnss_reading = deque_GNSS_readings.front();
        deque_GNSS_readings.pop_front();
      }

      am_hal_interrupt_master_enable();

      if (should_log_data){
        entry_kind[0] = '\n';
        entry_kind[1] = 'G';
        entry_kind[2] = 'P';
        entry_kind[3] = 'S';
        sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(entry_kind), sizeof(entry_kind));
        sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(&common_gnss_reading), sizeof(common_gnss_reading));
        should_log_data = false;
      }

      // with the IMU deque
      am_hal_interrupt_master_disable();

      if (deque_ACC_readings.size() > 0){
        should_log_data = true;
        common_acc_reading = deque_ACC_readings.front();
        deque_ACC_readings.pop_front();
      }

      am_hal_interrupt_master_enable();

      if (should_log_data){
        entry_kind[0] = '\n';
        entry_kind[1] = 'A';
        entry_kind[2] = 'C';
        entry_kind[3] = 'C';
        sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(entry_kind), sizeof(entry_kind));
        sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(&common_acc_reading), sizeof(common_acc_reading));
        should_log_data = false;
      }

      wdt.restart();
    }

    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(str_stop_logging), sizeof(str_stop_logging)-1);
    wdt.restart();
    // time to close the file and start logging a new one
    SERIAL_USB->println(F("Time to start new log file"));
    wdt.restart();
    sd_card_manager.close_and_sync_file();
    delay(10);
  }

  /////////////////////////////////////////////////////////////////////////////////
  // if we reach here, we have an issue:
  // stop SD card manager and let watchdog reset the board

  SERIAL_USB->println(F("Stopping logging due to error..."));
  sd_card_manager.close_and_sync_file();
  delay(1000);
  sd_card_manager.stop();

  SERIAL_USB->println(F("Entering infinite loop to trigger watchdog reset..."));

  while (true)
  {
    // reboot
    delay(1000);
  }

}

void loop() {
  // we should never get here
  // if we get here, the watchdog will eventually restart the board
}
