#ifndef FAST_LOGGER_H
#define FAST_LOGGER_H

#include "Arduino.h"

#include "watchdog_manager.h"

#include <SparkFun_u-blox_GNSS_v3.h>
#include <ISM330DHCXSensor.h>
#include "sd_card_manager.h"

class FastLogger {
    FastLogger(SFE_UBLOX_GNSS &gnss, ISM330DHCXSensor &accGyr, SD_Card_Manager &sdManager)
        : log_GNSS(gnss), AccGyr(accGyr), sd_card_manager(sdManager) {};
    
    void setup();

    void book_keep();

    SFE_UBLOX_GNSS &log_GNSS;
    ISM330DHCXSensor &AccGyr;
    SD_Card_Manager &sd_card_manager;

    bool sd_file_open = false;
};

#endif