"""The module for decoding the OLA ISM330DHXC + SAM-M10Q data files."""

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# Magic constants
HEADER_LINES_TO_READ = 10
PPS_MARKER = b"\nPPS"
GPS_MARKER = b"\nGPS"
IMU_MARKER = b"\nIMU"
FOOTER_MARKER = b"Log stop OLA"
MARKER_SIZE = 4
PPS_STRUCT_SIZE = 4
GPS_STRUCT_SIZE = 33
IMU_STRUCT_SIZE = 16
PPS_LINE_SIZE = MARKER_SIZE + PPS_STRUCT_SIZE  # 8 bytes
GPS_LINE_SIZE = MARKER_SIZE + GPS_STRUCT_SIZE  # 37 bytes
IMU_LINE_SIZE = MARKER_SIZE + IMU_STRUCT_SIZE  # 20 bytes


@dataclass
class PPSFix:
    """PPS fix data structure."""

    millis_reading: int


@dataclass
class GNSSReading:
    """GNSS reading data structure."""

    millis_reading: int
    latitude: int
    longitude: int
    posix_timestamp: int
    microseconds: int
    ned_vel_north: int
    ned_vel_east: int
    ned_vel_down: int
    fix_type: int
    latitude_dd: float
    longitude_dd: float
    ned_vel_north_mmps: int
    ned_vel_east_mmps: int
    ned_vel_down_mmps: int
    datetime_utc: datetime


@dataclass
class IMUReading:
    """IMU reading data structure."""

    millis_reading: int
    acc_x: int
    acc_y: int
    acc_z: int
    gyr_x: int
    gyr_y: int
    gyr_z: int
    acc_x_mg: float
    acc_y_mg: float
    acc_z_mg: float
    gyr_x_mdps: float
    gyr_y_mdps: float
    gyr_z_mdps: float


def parse_header(file_path: Path) -> dict[str, Any]:
    """Parse the header of the data file and extract metadata.

    Args:
        file_path: Path to the data file

    Returns:
        Dictionary containing parsed header information
    """
    header_info = {}

    with open(file_path, "rb") as f:
        header_lines = []
        for _ in range(HEADER_LINES_TO_READ):
            line = f.readline()
            if line:
                header_lines.append(line.decode("utf-8", errors="ignore"))

    for line in header_lines:
        if "ISM330DHCX Acc sensitivity" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["acc_sensitivity"] = float(parts[1].strip())
        elif "ISM330DHCX Gyr sensitivity" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["gyr_sensitivity"] = float(parts[1].strip())
        elif "ISM330DHCX ODR" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["imu_odr"] = float(parts[1].strip())
        elif "GNSS update rate" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["gnss_rate"] = float(parts[1].strip())
        elif "Firmware commit ID" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["firmware_commit"] = parts[1].strip()

    logger.info(f"Parsed header: {header_info}")
    return header_info


def parse_pps_entry(data: bytes) -> PPSFix:
    """Parse a single PPS entry.

    Args:
        data: Raw binary data for PPS entry

    Returns:
        PPSFix object

    Raises:
        AssertionError: If data size is incorrect
    """
    assert len(data) == PPS_STRUCT_SIZE, (
        f"PPS data size mismatch: expected exactly {PPS_STRUCT_SIZE} bytes, "
        f"got {len(data)} bytes"
    )
    millis_reading = struct.unpack("<I", data[:4])[0]
    return PPSFix(millis_reading=millis_reading)


def parse_gnss_entry(data: bytes) -> GNSSReading:
    """Parse a single GNSS entry.

    Args:
        data: Raw binary data for GNSS entry

    Returns:
        GNSSReading object with raw and physical unit values

    Raises:
        AssertionError: If data size is incorrect
    """
    assert len(data) == GPS_STRUCT_SIZE, (
        f"GNSS data size mismatch: expected exactly {GPS_STRUCT_SIZE} bytes, "
        f"got {len(data)} bytes"
    )
    values = struct.unpack("<IiiiIiiiB", data[:33])

    millis_reading = values[0]
    latitude = values[1]
    longitude = values[2]
    posix_timestamp = values[3]
    microseconds = values[4]
    ned_vel_north = values[5]
    ned_vel_east = values[6]
    ned_vel_down = values[7]
    fix_type = values[8]

    # Convert to physical units
    latitude_dd = latitude / 1e7
    longitude_dd = longitude / 1e7
    ned_vel_north_mmps = ned_vel_north
    ned_vel_east_mmps = ned_vel_east
    ned_vel_down_mmps = ned_vel_down

    # Create datetime with microsecond accuracy
    datetime_utc = datetime.fromtimestamp(
        posix_timestamp + microseconds / 1e6, tz=timezone.utc
    )

    return GNSSReading(
        millis_reading=millis_reading,
        latitude=latitude,
        longitude=longitude,
        posix_timestamp=posix_timestamp,
        microseconds=microseconds,
        ned_vel_north=ned_vel_north,
        ned_vel_east=ned_vel_east,
        ned_vel_down=ned_vel_down,
        fix_type=fix_type,
        latitude_dd=latitude_dd,
        longitude_dd=longitude_dd,
        ned_vel_north_mmps=ned_vel_north_mmps,
        ned_vel_east_mmps=ned_vel_east_mmps,
        ned_vel_down_mmps=ned_vel_down_mmps,
        datetime_utc=datetime_utc,
    )


def parse_imu_entry(
    data: bytes,
    acc_sensitivity: float = 0.061,
    gyr_sensitivity: float = 4.375,
) -> IMUReading:
    """Parse a single IMU entry.

    Args:
        data: Raw binary data for IMU entry
        acc_sensitivity: Accelerometer sensitivity in mg/LSB
        gyr_sensitivity: Gyroscope sensitivity in mdps/LSB

    Returns:
        IMUReading object with raw and scaled values

    Raises:
        AssertionError: If data size is incorrect
    """
    assert len(data) == IMU_STRUCT_SIZE, (
        f"IMU data size mismatch: expected exactly {IMU_STRUCT_SIZE} bytes, "
        f"got {len(data)} bytes"
    )
    values = struct.unpack("<Ihhhhhh", data[:16])
    millis_reading = values[0]
    acc_x, acc_y, acc_z = values[1], values[2], values[3]
    gyr_x, gyr_y, gyr_z = values[4], values[5], values[6]

    acc_x_mg = acc_x * acc_sensitivity
    acc_y_mg = acc_y * acc_sensitivity
    acc_z_mg = acc_z * acc_sensitivity
    gyr_x_mdps = gyr_x * gyr_sensitivity
    gyr_y_mdps = gyr_y * gyr_sensitivity
    gyr_z_mdps = gyr_z * gyr_sensitivity

    return IMUReading(
        millis_reading=millis_reading,
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        gyr_x=gyr_x,
        gyr_y=gyr_y,
        gyr_z=gyr_z,
        acc_x_mg=acc_x_mg,
        acc_y_mg=acc_y_mg,
        acc_z_mg=acc_z_mg,
        gyr_x_mdps=gyr_x_mdps,
        gyr_y_mdps=gyr_y_mdps,
        gyr_z_mdps=gyr_z_mdps,
    )


def print_summary_statistics(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    imu_list: list[IMUReading],
) -> None:
    """Print summary statistics about the parsed data.

    Args:
        pps_list: List of parsed PPS entries
        gnss_list: List of parsed GNSS entries
        imu_list: List of parsed IMU entries
    """
    logger.info("=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)

    logger.info("Number of messages parsed:")
    logger.info(f"  PPS:  {len(pps_list):6d}")
    logger.info(f"  GNSS: {len(gnss_list):6d}")
    logger.info(f"  IMU:  {len(imu_list):6d}")

    if len(imu_list) >= 2:
        first_millis = imu_list[0].millis_reading
        last_millis = imu_list[-1].millis_reading
        duration_ms = last_millis - first_millis
        duration_s = duration_ms / 1000.0
        duration_min = duration_s / 60.0

        logger.info("\nFile duration (from IMU timestamps):")
        logger.info(f"  First millis: {first_millis}")
        logger.info(f"  Last millis:  {last_millis}")
        logger.info(
            f"  Duration:     {duration_s:.2f} seconds "
            f"({duration_min:.2f} min)"
        )

        if duration_s > 0:
            logger.info("\nEffective sampling rates:")
            if len(pps_list) > 0:
                pps_rate = len(pps_list) / duration_s
                logger.info(f"  PPS:  {pps_rate:.3f} Hz")
            if len(gnss_list) > 0:
                gnss_rate = len(gnss_list) / duration_s
                logger.info(f"  GNSS: {gnss_rate:.3f} Hz")
            if len(imu_list) > 0:
                imu_rate = len(imu_list) / duration_s
                logger.info(f"  IMU:  {imu_rate:.3f} Hz")
    else:
        logger.warning("Not enough IMU data to compute duration")

    logger.info("=" * 60)


def decode_file(
    input_file: Path,
    output_dir: Path | None = None,
    pps_marker: bytes = PPS_MARKER,
    gps_marker: bytes = GPS_MARKER,
    imu_marker: bytes = IMU_MARKER,
    footer_marker: bytes = FOOTER_MARKER,
    pps_struct_size: int = PPS_STRUCT_SIZE,
    gps_struct_size: int = GPS_STRUCT_SIZE,
    imu_struct_size: int = IMU_STRUCT_SIZE,
) -> dict[str, Path]:
    """Decode a single data file and save to numpy arrays.

    Args:
        input_file: Path to input data file
        output_dir: Directory to save output files (defaults to same as input)
        pps_marker: Marker bytes for PPS entries
        gps_marker: Marker bytes for GPS entries
        imu_marker: Marker bytes for IMU entries
        footer_marker: Marker bytes for footer
        pps_struct_size: Size of PPS struct in bytes
        gps_struct_size: Size of GPS struct in bytes
        imu_struct_size: Size of IMU struct in bytes

    Returns:
        Dictionary with paths to saved numpy files
    """
    logger.info(f"Decoding file: {input_file}")

    if output_dir is None:
        output_dir = input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    header_info = parse_header(input_file)
    acc_sensitivity = header_info.get("acc_sensitivity", 0.061)
    gyr_sensitivity = header_info.get("gyr_sensitivity", 4.375)

    pps_list = []
    gnss_list = []
    imu_list = []

    with open(input_file, "rb") as f:
        content = f.read()

    idx = 0
    while idx < len(content):
        if content[idx : idx + 4] == pps_marker:
            # Check we have enough bytes for the full line
            line_end = idx + PPS_LINE_SIZE
            assert line_end <= len(content), (
                f"PPS line at offset {idx} incomplete: "
                f"need {PPS_LINE_SIZE} bytes, only {len(content) - idx} "
                f"bytes remaining"
            )

            idx += 4
            pps_data = content[idx : idx + pps_struct_size]
            try:
                pps_entry = parse_pps_entry(pps_data)
                pps_list.append(pps_entry)
            except (struct.error, AssertionError) as e:
                logger.error(
                    f"Failed to parse PPS entry at offset {idx}: {e}. "
                    f"Data length: {len(pps_data)}, expected: {pps_struct_size}"
                )
                raise
            idx += pps_struct_size

            # Assert next byte is \n (start of next entry), null padding, or footer
            if idx < len(content):
                next_byte = content[idx]
                assert (next_byte == ord(b'\n') or
                        next_byte == 0x00 or
                        footer_marker in content[idx:idx+20]), (
                    f"After PPS entry at offset {idx}: expected '\\n', null byte, "
                    f"or footer, got byte {next_byte:02x}"
                )

        elif content[idx : idx + 4] == gps_marker:
            # Check we have enough bytes for the full line
            line_end = idx + GPS_LINE_SIZE
            assert line_end <= len(content), (
                f"GNSS line at offset {idx} incomplete: "
                f"need {GPS_LINE_SIZE} bytes, only {len(content) - idx} "
                f"bytes remaining"
            )

            idx += 4
            gnss_data = content[idx : idx + gps_struct_size]
            try:
                gnss_entry = parse_gnss_entry(gnss_data)
                gnss_list.append(gnss_entry)
            except (struct.error, AssertionError) as e:
                logger.error(
                    f"Failed to parse GNSS entry at offset {idx}: {e}. "
                    f"Data length: {len(gnss_data)}, expected: {gps_struct_size}"
                )
                raise
            idx += gps_struct_size

            # Assert next byte is \n (start of next entry), null padding, or footer
            if idx < len(content):
                next_byte = content[idx]
                assert (next_byte == ord(b'\n') or
                        next_byte == 0x00 or
                        footer_marker in content[idx:idx+20]), (
                    f"After GNSS entry at offset {idx}: expected '\\n', null byte, "
                    f"or footer, got byte {next_byte:02x}"
                )

        elif content[idx : idx + 4] == imu_marker:
            # Check we have enough bytes for the full line
            line_end = idx + IMU_LINE_SIZE
            assert line_end <= len(content), (
                f"IMU line at offset {idx} incomplete: "
                f"need {IMU_LINE_SIZE} bytes, only {len(content) - idx} "
                f"bytes remaining"
            )

            idx += 4
            imu_data = content[idx : idx + imu_struct_size]
            try:
                imu_entry = parse_imu_entry(
                    imu_data, acc_sensitivity, gyr_sensitivity
                )
                imu_list.append(imu_entry)
            except (struct.error, AssertionError) as e:
                logger.error(
                    f"Failed to parse IMU entry at offset {idx}: {e}. "
                    f"Data length: {len(imu_data)}, expected: {imu_struct_size}"
                )
                raise
            idx += imu_struct_size

            # Assert next byte is \n (start of next entry), null padding, or footer
            if idx < len(content):
                next_byte = content[idx]
                assert (next_byte == ord(b'\n') or
                        next_byte == 0x00 or
                        footer_marker in content[idx:idx+20]), (
                    f"After IMU entry at offset {idx}: expected '\\n', null byte, "
                    f"or footer, got byte {next_byte:02x}"
                )

        elif footer_marker in content[idx : idx + len(footer_marker) + 10]:
            logger.info("Found footer marker, stopping parsing")
            break
        else:
            idx += 1

    logger.info(
        f"Parsed {len(pps_list)} PPS, {len(gnss_list)} GNSS, {len(imu_list)} IMU"
    )

    # Print summary statistics
    print_summary_statistics(pps_list, gnss_list, imu_list)

    base_name = input_file.stem
    output_files = {}

    pps_array = np.array(pps_list, dtype=object)
    pps_file = output_dir / f"{base_name}_pps.npy"
    np.save(pps_file, pps_array)
    output_files["pps"] = pps_file
    logger.info(f"Saved PPS data to {pps_file}")

    gnss_array = np.array(gnss_list, dtype=object)
    gnss_file = output_dir / f"{base_name}_gnss.npy"
    np.save(gnss_file, gnss_array)
    output_files["gnss"] = gnss_file
    logger.info(f"Saved GNSS data to {gnss_file}")

    imu_array = np.array(imu_list, dtype=object)
    imu_file = output_dir / f"{base_name}_imu.npy"
    np.save(imu_file, imu_array)
    output_files["imu"] = imu_file
    logger.info(f"Saved IMU data to {imu_file}")

    return output_files
