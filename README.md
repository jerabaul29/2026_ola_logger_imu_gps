# 2026_ola_logger_imu_gps: OLA Waves Logger (OWL)

An OLA logger for IMU + GPS measurements to an SD card.

See https://github.com/jerabaul29/2026_ola_logger_imu_gps/tree/main/logger_ola_imu_gps for the description of the hardware + the firmware.

See https://github.com/jerabaul29/2026_ola_logger_imu_gps/tree/main/decoder for tooling around binary log files data extraction.

---

I wrote the firmware as a team together with help from Copilot in PlatformIO + VSC.

I wrote the decoder largely using Copilot CLI, only writing down the intent in the SPEC.md and AGENT.md, and letting Copilot CLI build the actual implementation with just a couple of prompt interactions with it.

---

If you use this design or a closely related "descendent" of it, consider referring to our preprint:

- title: Open Wave Logger v2026 (OWL-v2026): an open source, low cost, easy to build, high performance logger for wave data measurements
- authors: Jean Rabault, Joey Voermans, Takuji Waseda, Takehiko Nose, Tsubasa Kodaira, Koya Sato, Alexander Babanin, Gaute Hope, Malte Müller, Lars Willas Dreyer, Øystein Lande, Atle Jensen, Øyvind Breivik
- year: 2026
- link: https://www.researchgate.net/publication/404067196_OPEN_WAVE_LOGGER_V2026_OWL-V2026_AN_OPEN_SOURCE_LOW_COST_EASY_TO_BUILD_HIGH_PERFORMANCE_LOGGER_FOR_WAVE_DATA_MEASUREMENTS
