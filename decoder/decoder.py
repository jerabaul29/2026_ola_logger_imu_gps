"""The module for decoding the OLA ISM330DHXC + SAM-M10Q data files."""

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats
import gnuplotlib as gp

# Magic constants
HEADER_SEARCH_BYTES = 64 * 1024
PPS_MARKER = b"\nPPS"
GPS_MARKER = b"\nGPS"
IMU_MARKER = b"\nIMU"
FOOTER_MARKER = b"Log stop OLA"
MARKER_SIZE = 4
PPS_STRUCT_SIZE = 4
GPS_STRUCT_SIZE = 36
IMU_STRUCT_SIZE = 16
PPS_LINE_SIZE = MARKER_SIZE + PPS_STRUCT_SIZE  # 8 bytes
GPS_LINE_SIZE = MARKER_SIZE + GPS_STRUCT_SIZE  # 40 bytes
IMU_LINE_SIZE = MARKER_SIZE + IMU_STRUCT_SIZE  # 20 bytes


@dataclass
class PPSFix:
    """PPS fix data structure."""

    millis_reading: int
    utc_timestamp_from_pps_regression: float | None = None
    datetime_timestamp_from_pps_regression: datetime | None = None


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
    utc_timestamp_from_pps_regression: float | None = None
    datetime_timestamp_from_pps_regression: datetime | None = None


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
    utc_timestamp_from_pps_regression: float | None = None
    datetime_timestamp_from_pps_regression: datetime | None = None


def parse_header(
    file_path: Path,
    markers: tuple[bytes, bytes, bytes] = (PPS_MARKER, GPS_MARKER, IMU_MARKER),
    search_bytes: int = HEADER_SEARCH_BYTES,
) -> dict[str, Any]:
    """Parse the header of the data file and extract metadata.

    Args:
        file_path: Path to the data file
        markers: Tuple of markers that indicate start of data section
        search_bytes: Number of bytes to scan from start of file for header

    Returns:
        Dictionary containing parsed header information
    """
    header_info = {}

    with open(file_path, "rb") as f:
        content = f.read(search_bytes)

    marker_positions = [
        pos for m in markers if (pos := content.find(m)) != -1
    ]
    header_end = min(marker_positions) if marker_positions else len(content)
    header_text = content[:header_end].decode("utf-8", errors="ignore")
    header_lines = header_text.splitlines()

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


def unwrap_millis(millis_list: list[int]) -> list[int]:
    """Unwrap uint32_t millis timestamps to handle wrapping.

    Since millis() uses uint32_t on the OLA MCU, we need to handle wrapping
    at 2**32. If a jump of more than 2**32/2 is detected going backwards,
    we add 2**32 to all following values.

    Args:
        millis_list: List of millis timestamps (may have wrapping)

    Returns:
        List of unwrapped millis timestamps
    """
    if not millis_list:
        return []

    UINT32_MAX = 2**32
    HALF_UINT32 = UINT32_MAX // 2

    unwrapped = [millis_list[0]]
    offset = 0

    for i in range(1, len(millis_list)):
        current = millis_list[i]
        prev = millis_list[i - 1]

        # Detect wrap: if current is much less than prev (jumped backwards)
        if prev - current > HALF_UINT32:
            offset += UINT32_MAX

        unwrapped.append(current + offset)

    return unwrapped


def compute_pps_regression(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    global_min_millis: int | None = None,
) -> tuple[float, float] | None:
    """Compute linear regression from PPS millis to UTC timestamps.

    This function:
    1. Unwraps millis timestamps for both PPS and GNSS data
    2. Matches each PPS timestamp with the closest GNSS timestamp
    3. Uses GNSS UTC time to determine which second each PPS corresponds to
    4. Performs linear regression to map millis -> UTC posix timestamp

    Args:
        pps_list: List of PPS fixes
        gnss_list: List of GNSS readings
        global_min_millis: Minimum millis value across all data types (for offset)

    Returns:
        Tuple of (slope, intercept) for the linear regression, or None if
        insufficient data
    """
    if not pps_list or not gnss_list:
        logger.warning("Cannot compute PPS regression: empty PPS or GNSS data")
        return None

    if len(pps_list) < 2:
        logger.warning(
            f"Cannot compute PPS regression: need at least 2 PPS entries, "
            f"got {len(pps_list)}"
        )
        return None

    # Unwrap millis for both PPS and GNSS
    pps_millis_raw = [p.millis_reading for p in pps_list]
    gnss_millis_raw = [g.millis_reading for g in gnss_list]

    pps_millis_unwrapped = unwrap_millis(pps_millis_raw)
    gnss_millis_unwrapped = unwrap_millis(gnss_millis_raw)

    # For each PPS entry, find the closest GNSS entry by millis
    # and determine which UTC second boundary the PPS marks
    pps_matched_millis = []
    pps_matched_utc = []

    for pps_millis in pps_millis_unwrapped:
        # Find closest GNSS entry
        min_diff = float("inf")
        closest_gnss_idx = 0

        for j, gnss_millis in enumerate(gnss_millis_unwrapped):
            diff = abs(gnss_millis - pps_millis)
            if diff < min_diff:
                min_diff = diff
                closest_gnss_idx = j

        # Get the UTC timestamp from the matched GNSS entry
        gnss_entry = gnss_list[closest_gnss_idx]
        utc_timestamp = gnss_entry.posix_timestamp + gnss_entry.microseconds / 1e6
        
        # Determine which second boundary this PPS marks
        # The PPS marks the start of a second. We estimate which second
        # by looking at the UTC time of the closest GNSS and the millis offset
        millis_offset = pps_millis - gnss_millis_unwrapped[closest_gnss_idx]
        estimated_pps_utc = utc_timestamp + millis_offset / 1000.0
        
        # The PPS second is the second boundary closest to the estimated time
        utc_second = round(estimated_pps_utc)

        pps_matched_millis.append(pps_millis)
        pps_matched_utc.append(float(utc_second))

    # Perform linear regression
    # To avoid numerical inaccuracies, subtract the minimum millis value
    # Use global minimum if provided, otherwise use minimum from PPS data
    if global_min_millis is None:
        min_millis = min(pps_matched_millis)
    else:
        min_millis = global_min_millis

    pps_matched_millis_offset = [m - min_millis for m in pps_matched_millis]

    slope, intercept_offset, r_value, p_value, std_err = stats.linregress(
        pps_matched_millis_offset, pps_matched_utc
    )

    # Adjust intercept to account for the offset we subtracted
    intercept = intercept_offset - slope * min_millis

    logger.info(f"PPS regression: slope={slope:.12f}, intercept={intercept:.6f}")
    logger.info(f"  R²={r_value**2:.9f}, p-value={p_value:.2e}, std_err={std_err:.2e}")
    logger.info(f"  Used {len(pps_matched_millis)} PPS-GNSS matched pairs")

    return (slope, intercept)


def apply_pps_regression(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    imu_list: list[IMUReading],
    slope: float,
    intercept: float,
) -> None:
    """Apply PPS regression to all data entries.

    This modifies the dataclass objects in-place, adding the
    utc_timestamp_from_pps_regression and datetime_timestamp_from_pps_regression fields.

    Args:
        pps_list: List of PPS fixes
        gnss_list: List of GNSS readings
        imu_list: List of IMU readings
        slope: Regression slope
        intercept: Regression intercept
    """
    # Unwrap and apply to PPS
    if pps_list:
        pps_millis_unwrapped = unwrap_millis([p.millis_reading for p in pps_list])
        for i, pps in enumerate(pps_list):
            pps.utc_timestamp_from_pps_regression = (
                slope * pps_millis_unwrapped[i] + intercept
            )
            pps.datetime_timestamp_from_pps_regression = datetime.fromtimestamp(
                pps.utc_timestamp_from_pps_regression, tz=timezone.utc
            )

    # Unwrap and apply to GNSS
    if gnss_list:
        gnss_millis_unwrapped = unwrap_millis([g.millis_reading for g in gnss_list])
        for i, gnss in enumerate(gnss_list):
            gnss.utc_timestamp_from_pps_regression = (
                slope * gnss_millis_unwrapped[i] + intercept
            )
            gnss.datetime_timestamp_from_pps_regression = datetime.fromtimestamp(
                gnss.utc_timestamp_from_pps_regression, tz=timezone.utc
            )

    # Unwrap and apply to IMU
    if imu_list:
        imu_millis_unwrapped = unwrap_millis([imu.millis_reading for imu in imu_list])
        for i, imu in enumerate(imu_list):
            imu.utc_timestamp_from_pps_regression = (
                slope * imu_millis_unwrapped[i] + intercept
            )
            imu.datetime_timestamp_from_pps_regression = datetime.fromtimestamp(
                imu.utc_timestamp_from_pps_regression, tz=timezone.utc
            )


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
        f"GNSS data size mismatch: expected exactly {GPS_STRUCT_SIZE} bytes "
        f"(33 bytes struct + 3 bytes padding), got {len(data)} bytes"
    )
    # Unpack first 33 bytes (actual struct), ignore 3 bytes padding
    data_to_unpack = data[:33]
    values = struct.unpack("<IiiiIiiiB", data_to_unpack)

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


def compute_pps_mismatch_statistics(pps_list: list[PPSFix]) -> None:
    """Compute and display PPS mismatch statistics and plot.
    
    For each PPS entry, computes the mismatch between the UTC datetime 
    from linear regression and the closest UTC second. Displays statistics
    and an ASCII terminal plot showing how the mismatch varies over time.
    
    Args:
        pps_list: List of PPS fixes with regression timestamps computed
    """
    if not pps_list:
        logger.warning("No PPS data available for mismatch analysis")
        return
    
    # Check if regression was computed
    if pps_list[0].utc_timestamp_from_pps_regression is None:
        logger.warning("PPS regression not computed, skipping mismatch analysis")
        return
    
    # Compute mismatch for each PPS entry
    mismatches = []
    for pps in pps_list:
        utc_timestamp = pps.utc_timestamp_from_pps_regression
        closest_second = round(utc_timestamp)
        mismatch = utc_timestamp - closest_second
        mismatches.append(mismatch)
    
    mismatches_array = np.array(mismatches)
    
    # Compute statistics
    max_mismatch = np.max(np.abs(mismatches_array))
    mean_mismatch = np.mean(mismatches_array)
    rms_mismatch = np.sqrt(np.mean(mismatches_array ** 2))
    
    # Log statistics
    logger.info("")
    logger.info("PPS Mismatch Statistics (UTC regression vs closest second):")
    logger.info(f"  Max absolute mismatch: {max_mismatch * 1000:.3f} ms")
    logger.info(f"  Mean mismatch:         {mean_mismatch * 1000:.3f} ms")
    logger.info(f"  RMS mismatch:          {rms_mismatch * 1000:.3f} ms")
    
    # Prepare data for plotting
    # X-axis: time since first PPS (in seconds)
    first_pps_utc = pps_list[0].utc_timestamp_from_pps_regression
    x_data = np.array([
        pps.utc_timestamp_from_pps_regression - first_pps_utc
        for pps in pps_list
    ])
    
    # Y-axis: mismatch in milliseconds
    y_data = mismatches_array * 1000
    
    # Plot using gnuplotlib
    import sys
    # Temporarily redirect stderr to stdout to ensure plot appears correctly
    old_stderr = sys.stderr
    sys.stderr = sys.stdout
    
    print("")  # Print blank line directly to stdout
    print("PPS Mismatch vs Time Plot:")
    sys.stdout.flush()
    gp.plot(
        x_data, y_data,
        _with='lines',
        terminal='dumb 80,24',
        unset='grid',
        xlabel='Time since first PPS (seconds)',
        ylabel='UTC mismatch (ms)',
        title='PPS UTC Mismatch (ms): Regression vs Closest Second'
    )
    print("")  # Print blank line after plot
    
    # Restore stderr
    sys.stderr = old_stderr


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

        logger.info("")
        logger.info("File duration (from IMU timestamps):")
        logger.info(f"  First millis: {first_millis}")
        logger.info(f"  Last millis:  {last_millis}")
        logger.info(
            f"  Duration:     {duration_s:.2f} seconds "
            f"({duration_min:.2f} min)"
        )

        if duration_s > 0:
            logger.info("")
            logger.info("Effective sampling rates:")
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

    header_info = parse_header(input_file, markers=(pps_marker, gps_marker, imu_marker))
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

    # Compute and apply PPS regression
    # Find global minimum millis across all data types
    all_millis = []
    if pps_list:
        all_millis.extend([p.millis_reading for p in pps_list])
    if gnss_list:
        all_millis.extend([g.millis_reading for g in gnss_list])
    if imu_list:
        all_millis.extend([i.millis_reading for i in imu_list])

    global_min_millis = min(all_millis) if all_millis else 0
    logger.info(f"Global minimum millis reading: {global_min_millis}")

    logger.info("Computing PPS to UTC timestamp regression...")
    regression = compute_pps_regression(pps_list, gnss_list, global_min_millis)
    if regression is None:
        logger.warning("Skipping PPS regression due to insufficient data")
    else:
        slope, intercept = regression
        apply_pps_regression(pps_list, gnss_list, imu_list, slope, intercept)
        
        # Compute and display PPS mismatch statistics
        compute_pps_mismatch_statistics(pps_list)

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
