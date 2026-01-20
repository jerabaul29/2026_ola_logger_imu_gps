# OLA Waves Logger (OWL)

## Hardware and assembly / connections

This is the code for a data logger with the following properties:

- main board: SF OLA (Sparkfun OpenLogArtemis) without built-in IMU
- ISM330DHCX over qwiic IMU (for example, the Sparkfun one)
- SAM-M10Q over qwiic + PPS interrupt (connect PPS GPS pin to pin 11 OLA) (for example, the Sparkfun one)

In addition, users may want to design a robust power supply, such as:

- batteries: SAFT LSH20 or similar to be power effective in cold conditions
- regulator: Pololu step up / step down to ensure stable 3.3V supply
- connect the 3.3V regulated power to the 3V3 OLA pin
- connect the GND regulated power to the GND OLA pin

In order to reach lowest possible power consumption, users may consider to cut the power LED pads on the back of boards that support it (such as the Sparkfun GPS and IMU breakout boards).

Use a good quality SD card to enable power efficient and fast logging.

## Notes about the design

The aim is to have a robust, high accuracy, high frequency, low jitter logger.

The ISM330DHCX (MEMS 3-axis accelerometer + 3-axis gyroscope) is logged through a timer driven interrupt routine and deque buffers at 417Hz by default. This ensures that the data are gathered reliably even when the CPU is busy with async tasks (e.g. writing to SD card). The IMU is set up to work at the highest possible accuracy level.

The GPS is logged through the same timer driven interrupt at 10Hz by default.

The UTC PPS seconds starts are logged through a rising edge interrupt acquired from the GPS PPS pin.

Writing to SD cards is performed asynchronously through a busy loop.

With this design, all the sensors logging is done through interrupts and buffered to deques, and all SD card writing is done asynchronously from the logging through a busy loop - this should result in reliable low jitter logging.

The power consumption is around XXmA (may depend on SD card used, satellite signal quality).

The amount of data generated is 1 file every 15 minutes, typical file size around XX KBytes.

## LED understanding

The red LED may blink during setup.

The blue LED should blink rapidly during logging (it is on while busy writing to SD, off when no ongoing SD writing).

The PPS LED should blink at 1Hz when GPS signal is available.

## Compiling / Uploading

The project uses the Sparkfun Artemis Arduino core v1, made available through the PlatformIO platform: see instructions at: https://github.com/nigelb/platform-apollo3blue . Make sure to choose Core V1. All dependencies are hard copied in the lib folder.

In addition we provide the .bin file, that has been compiled in advance and can be uploaded directly to the OLA using: https://github.com/sparkfun/Artemis-Firmware-Upload-GUI .

## Data extraction from files

The data on the SD card are stored largely in binary format for efficiency (otherwise, the bandwidth to be written to SD card would be too large, and the SD card would also fill too fast). See the decoder in the adjacent folder to extract the data from the SD card.

## Disclaimers

I wanted to make this into a "clean" project, but I ended up time constrained, so this is a mix of old libs, new libs, custom libs, and various stuff I had from other projects around the years - this is a bit messy!