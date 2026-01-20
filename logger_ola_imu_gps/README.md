# OLA Waves Logger (OWL)

## Hardware and assembly / connections

This is the code for a data logger with the following properties:

- main board: SF OLA (Sparkfun OpenLogArtemis) without built-in IMU
- ISM330DHCX over qwiic IMU (for example, the Sparkfun one)
- SAM-M10Q over qwiic + PPS interrupt (connect PPS GPS pin to pin 11 OLA) (for example, the Sparkfun one)

I recommend going OLA <-> ISM330DHCX breakout <-> GPS breakout, and keep wires relatively short. The I2C bus is run at 400kHz.

In addition, users may want to design a robust power supply, such as:

- batteries: SAFT LSH20 or similar to be power effective in cold conditions
- regulator: Pololu step up / step down to ensure stable 3.3V supply
- connect the 3.3V regulated power to the 3V3 OLA pin
- connect the GND regulated power to the GND OLA pin

In order to reach lowest possible power consumption, users may consider to cut the power LED pads on the back of boards that support it (such as the Sparkfun GPS and IMU breakout boards).

Use a good quality SD card to enable power efficient and fast logging. The SD card needs to be formatted as FAT before use.

## Notes about the design

The aim is to have a robust, high accuracy, high frequency, low jitter logger.

The ISM330DHCX (MEMS 3-axis accelerometer + 3-axis gyroscope) is logged through a timer driven interrupt routine and deque buffers at 417Hz by default. This ensures that the data are gathered reliably even when the CPU is busy with async tasks (e.g. writing to SD card). The IMU is set up to work at the highest possible accuracy level.

The GPS is logged through the same timer driven interrupt at 10Hz by default.

The UTC PPS seconds starts are logged through a rising edge interrupt acquired from the GPS PPS pin.

Writing to SD cards is performed asynchronously through a busy loop.

The full sketch is under watchdog timer control, so the board will hard reboot in case of an issue.

With this design, all the sensors logging is done through interrupts and buffered to deques, and all SD card writing is done asynchronously from the logging through a busy loop - this should result in reliable low jitter logging.

The power consumption is around XXmA (may depend on SD card used, satellite signal quality).

The amount of data generated is 1 file every 15 minutes, typical file size around XX KBytes.

## LED understanding

The red LED may blink during setup.

The blue LED should blink rapidly / flicker during logging (it is on while busy writing to SD, off when no ongoing SD writing).

The PPS LED should blink at 1Hz when GPS signal is available.

The logger only logs if it gets GPS information - if no GPS information within 5 minutes, it will go to sleep for 1 hour to save battery before trying again. Press the reset button if you want to force restart the firmware.

## Compiling / Uploading

The project uses the Sparkfun Artemis Arduino core v1, made available through the PlatformIO platform: see instructions at: https://github.com/nigelb/platform-apollo3blue . Make sure to choose Core V1. All dependencies are hard copied in the lib folder.

In addition we provide the .bin file, that has been compiled in advance and can be uploaded directly to the OLA using: https://github.com/sparkfun/Artemis-Firmware-Upload-GUI .

## Data extraction from files

The data on the SD card are stored largely in binary format for efficiency (otherwise, the bandwidth to be written to SD card would be too large, and the SD card would also fill too fast). See the decoder in the adjacent folder to extract the data from the SD card.

## Choice of SD card

SD cards come in many different qualities, and speed and power consumption can vary a lot. Choose a "reasonably good" one, otherwise power consumption may be bad and / or writing speed too slow compared to the data stream. Choosing a good SD card is a dark art, you may need to test with different models!

## Disclaimers

I wanted to make this into a "clean" project, but I ended up time constrained, so this is a mix of old libs, new libs, custom libs, and various stuff I had from other projects around the years - this is a bit messy!

## Serial logs

When running the logger while connect to USB, some level of information / log is provided over USB (serial baudrate 1 million: 1000000). For example:

```
millis(): 1539116; seconds since boot: 1539
Samples logged in last interval: IMU: 4443; GNSS: 95; PPS: 8
Max deque sizes reached: IMU: 21 over 12510; ISM FIFO: 12 over 512; GNSS: 1 over 300; PPS: 1 over 30
Effective logging rates (Hz): IMU (Hz): 444.20; GNSS (Hz): 9.50; PPS (Hz): 0.80
Accumulated SD time (ms): 4810 ms over 10000 ms interval

millis(): 1549127; seconds since boot: 1549
Samples logged in last interval: IMU: 4432; GNSS: 96; PPS: 9
Max deque sizes reached: IMU: 20 over 12510; ISM FIFO: 10 over 512; GNSS: 1 over 300; PPS: 1 over 30
Effective logging rates (Hz): IMU (Hz): 443.10; GNSS (Hz): 9.60; PPS (Hz): 0.90
Accumulated SD time (ms): 4822 ms over 10000 ms interval

Time to start new log file
Final log file size (KBytes): 10240
Final log file available size remaining (KBytes): 2147
Syncing and closing file...
File closed

Preparing to start new log file...
Opening file: DATA_BOOT_0058_TIME_20260120T220000.dat
File opened successfully
Preallocating 10485760 bytes...
File preallocated successfully
Log file opened and preallocated successfully.
Current posix timestamp: 1768946402
Next log file posix timestamp: 1768947300
Logging...

Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: f46f8cf748345997a2077f094a2bc248dbabcaa5
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10

millis(): 1563287; seconds since boot: 1563
Samples logged in last interval: IMU: 4433; GNSS: 93; PPS: 10
Max deque sizes reached: IMU: 1255 over 12510; ISM FIFO: 10 over 512; GNSS: 27 over 300; PPS: 2 over 30
Effective logging rates (Hz): IMU (Hz): 443.20; GNSS (Hz): 9.30; PPS (Hz): 1.00
Accumulated SD time (ms): 6406 ms over 10000 ms interval

millis(): 1573308; seconds since boot: 1573
Samples logged in last interval: IMU: 4437; GNSS: 96; PPS: 10
Max deque sizes reached: IMU: 19 over 12510; ISM FIFO: 10 over 512; GNSS: 1 over 300; PPS: 1 over 30
Effective logging rates (Hz): IMU (Hz): 443.50; GNSS (Hz): 9.60; PPS (Hz): 1.00
Accumulated SD time (ms): 4885 ms over 10000 ms interval
```