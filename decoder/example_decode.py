"""Example script demonstrating how to decode a data file."""

import sys
from pathlib import Path

import numpy as np
from loguru import logger

from decoder import decode_file

if __name__ == "__main__":
    data_file = Path("DATA_BOOT_0055_TIME_20260120T211500.dat")

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)

    logger.info(f"Decoding data file: {data_file}")
    output_files = decode_file(data_file)

    logger.info("Loading decoded data...")

    pps_data = np.load(output_files["pps"], allow_pickle=True)
    logger.info(f"Loaded {len(pps_data)} PPS entries")
    if len(pps_data) > 0:
        logger.info(f"First PPS entry: {pps_data[0]}")

    gnss_data = np.load(output_files["gnss"], allow_pickle=True)
    logger.info(f"Loaded {len(gnss_data)} GNSS entries")
    if len(gnss_data) > 0:
        logger.info(f"First GNSS entry: {gnss_data[0]}")

    imu_data = np.load(output_files["imu"], allow_pickle=True)
    logger.info(f"Loaded {len(imu_data)} IMU entries")
    if len(imu_data) > 0:
        logger.info(f"First IMU entry: {imu_data[0]}")

    logger.success("Decoding completed successfully!")
