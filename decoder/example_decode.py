"""Example script demonstrating how to decode a data file."""

import sys
from pathlib import Path

import numpy as np
from loguru import logger

from decoder import decode_file


def print_data_sample(data, data_type: str, n: int = 10) -> None:
    """Print first n and last n entries of data array.
    
    Args:
        data: Numpy array of data entries
        data_type: Type name for logging (e.g., "PPS", "GNSS", "IMU")
        n: Number of entries to show from start and end
    """
    if len(data) == 0:
        logger.warning(f"No {data_type} data to display")
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"{data_type} DATA SAMPLE - Loaded {len(data)} entries")
    logger.info("=" * 80)

    # Show first n entries
    n_first = min(n, len(data))
    logger.info(f"First {n_first} entries:")
    for i in range(n_first):
        logger.info(f"  [{i}] {data[i]}")

    # Show last n entries if we have more than n entries
    if len(data) > n:
        logger.info("")
        n_last = min(n, len(data))
        logger.info(f"Last {n_last} entries:")
        for i in range(len(data) - n_last, len(data)):
            logger.info(f"  [{i}] {data[i]}")


if __name__ == "__main__":
    data_file = Path("DATA_BOOT_0000_TIME_20260204T193000.dat")

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)

    logger.info(f"Decoding data file: {data_file}")
    # To enable ASCII plots, set show_plots=True
    output_files = decode_file(data_file, show_plots=False)

    logger.info("")
    logger.info("=" * 80)
    logger.info("LOADING DECODED DATA")
    logger.info("=" * 80)

    # Load the decoded data from compressed archive
    npz_file = output_files["file"]
    with np.load(npz_file, allow_pickle=True) as data:
        pps_data = data["pps"]
        gnss_data = data["gnss"]
        imu_data = data["imu"]
        
        # Display header information
        logger.info("")
        logger.info("=" * 80)
        logger.info("HEADER INFORMATION")
        logger.info("=" * 80)
        if "header_string" in data:
            logger.info(f"Full header:\n{data['header_string'][0]}")
        if "acc_sensitivity" in data:
            logger.info(f"Acc sensitivity: {data['acc_sensitivity'][0]} mg/LSB")
        if "gyr_sensitivity" in data:
            logger.info(f"Gyr sensitivity: {data['gyr_sensitivity'][0]} mdps/LSB")
        if "imu_odr" in data:
            logger.info(f"IMU ODR: {data['imu_odr'][0]} Hz")
        if "gnss_rate" in data:
            logger.info(f"GNSS rate: {data['gnss_rate'][0]} Hz")
        if "firmware_commit" in data:
            logger.info(f"Firmware commit: {data['firmware_commit'][0]}")

    print_data_sample(pps_data, "PPS", n=10)
    print_data_sample(gnss_data, "GNSS", n=10)
    print_data_sample(imu_data, "IMU", n=10)

    logger.info("")
    logger.success("Decoding completed successfully!")
