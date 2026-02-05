"""The module for decoding the OLA ISM330DHXC + SAM-M10Q data files."""

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats

# Try to import gnuplotlib, but don't fail if not available
try:
    import gnuplotlib as gp
    GNUPLOT_AVAILABLE = True
except ImportError:
    GNUPLOT_AVAILABLE = False
    logger.warning(
        "gnuplotlib not available - plots will be disabled. "
        "Install with: pip install gnuplotlib"
    )

# Magic constants
HEADER_SEARCH_BYTES = 64 * 1024
PPS_MARKER = b"\nPPS"
GPS_MARKER = b"\nGPS"
IMU_MARKER = b"\nIMU"
FOOTER_MARKER = b"Log stop OLA"
MARKER_SIZE = 4
PPS_STRUCT_SIZE = 4
GPS_STRUCT_SIZE = 36
IMU_STRUCT_SIZE = 18
IMU_PADDING = 2  # C struct alignment padding
PPS_LINE_SIZE = MARKER_SIZE + PPS_STRUCT_SIZE  # 8 bytes
GPS_LINE_SIZE = MARKER_SIZE + GPS_STRUCT_SIZE  # 40 bytes
IMU_LINE_SIZE = MARKER_SIZE + IMU_STRUCT_SIZE + IMU_PADDING  # 24 bytes (4 + 18 + 2)


@dataclass
class PPSFix:
    """PPS fix data structure."""

    micros_reading: int
    micros_reading_unwrapped: int | None = None
    utc_timestamp_from_pps_regression: float | None = None
    datetime_timestamp_from_pps_regression: datetime | None = None


@dataclass
class GNSSReading:
    """GNSS reading data structure."""

    micros_reading: int
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
    micros_reading_unwrapped: int | None = None
    utc_timestamp_from_pps_regression: float | None = None
    datetime_timestamp_from_pps_regression: datetime | None = None


@dataclass
class IMUReading:
    """IMU reading data structure."""

    micros_reading: int
    counter: int
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
    micros_reading_unwrapped: int | None = None
    counter_unwrapped: int | None = None
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


def unwrap_array(
    values: np.ndarray,
    max_value: int,
    wrap_threshold: float | None = None,
    jump_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Unwrap potentially wrapping array and detect anomalous jumps.

    This function:
    1. Detects wraps: large negative jumps near max_value boundary
    2. Unwraps: applies offset corrections for wrapped values
    3. Detects jumps on unwrapped data:
       - Any negative jump (data should be monotonic after unwrapping)
       - Any positive jump > jump_threshold

    Args:
        values: Array of potentially wrapping values (e.g., micros, counter)
        max_value: Maximum value before wrapping (e.g., 2**32 for uint32)
        wrap_threshold: Threshold for wrap detection (default: 0.75 * max_value)
        jump_threshold: Threshold for jump detection (default: 0.1 * max_value)

    Returns:
        Tuple of:
        - unwrapped_array: Array with wrapping corrected
        - wrap_indices: Indices where wraps occurred (or None if no wraps)
        - jump_indices: Indices where anomalous jumps occurred (or None if no jumps)
    """
    if len(values) == 0:
        return np.array([]), None, None

    if wrap_threshold is None:
        wrap_threshold = 0.75 * max_value
    if jump_threshold is None:
        jump_threshold = 0.1 * max_value

    # Step 1: Detect wraps and unwrap
    unwrapped = np.zeros_like(values, dtype=np.int64)
    unwrapped[0] = values[0]
    offset = 0
    wrap_indices_list = []

    for i in range(1, len(values)):
        current = values[i]
        prev = values[i - 1]
        diff = current - prev

        # Detect wrap: large negative jump
        if diff < 0 and abs(diff) > wrap_threshold:
            offset += max_value
            wrap_indices_list.append(i)

        unwrapped[i] = current + offset

    # Step 2: Detect anomalous jumps on unwrapped data
    jump_indices_list = []

    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i - 1]

        # Any negative jump is anomalous (should be monotonic after unwrapping)
        # Any positive jump > threshold is anomalous
        if diff < 0 or diff > jump_threshold:
            jump_indices_list.append(i)

    # Convert to numpy arrays or None
    wrap_indices = (
        np.array(wrap_indices_list, dtype=np.int64)
        if wrap_indices_list
        else None
    )
    jump_indices = (
        np.array(jump_indices_list, dtype=np.int64)
        if jump_indices_list
        else None
    )

    return unwrapped, wrap_indices, jump_indices


def compute_pps_regression(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    global_min_micros: int | None = None,
) -> tuple[float, float] | None:
    """Compute linear regression from PPS micros to UTC timestamps.

    This function:
    1. Uses unwrapped micros timestamps for both PPS and GNSS data
    2. Matches each PPS timestamp with the closest GNSS timestamp
    3. Uses GNSS UTC time to determine which second each PPS corresponds to
    4. Performs linear regression to map micros -> UTC posix timestamp

    Args:
        pps_list: List of PPS fixes (must have micros_reading_unwrapped populated)
        gnss_list: List of GNSS readings (must have micros_reading_unwrapped populated)
        global_min_micros: Minimum micros value across all data types (for offset)

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

    # Get unwrapped micros for both PPS and GNSS
    pps_micros_unwrapped = [
        p.micros_reading_unwrapped if p.micros_reading_unwrapped is not None
        else p.micros_reading
        for p in pps_list
    ]
    gnss_micros_unwrapped = [
        g.micros_reading_unwrapped if g.micros_reading_unwrapped is not None
        else g.micros_reading
        for g in gnss_list
    ]

    # For each PPS entry, find the closest GNSS entry by micros
    # and determine which UTC second boundary the PPS marks
    pps_matched_micros = []
    pps_matched_utc = []

    for pps_micros in pps_micros_unwrapped:
        # Find closest GNSS entry
        min_diff = float("inf")
        closest_gnss_idx = 0

        for j, gnss_micros in enumerate(gnss_micros_unwrapped):
            diff = abs(gnss_micros - pps_micros)
            if diff < min_diff:
                min_diff = diff
                closest_gnss_idx = j

        # Get the UTC timestamp from the matched GNSS entry
        gnss_entry = gnss_list[closest_gnss_idx]
        utc_timestamp = gnss_entry.posix_timestamp + gnss_entry.microseconds / 1e6

        # Determine which second boundary this PPS marks
        # The PPS marks the start of a second. We estimate which second
        # by looking at the UTC time of the closest GNSS and the micros offset
        micros_offset = pps_micros - gnss_micros_unwrapped[closest_gnss_idx]
        estimated_pps_utc = utc_timestamp + micros_offset / 1e6

        # The PPS second is the second boundary closest to the estimated time
        utc_second = round(estimated_pps_utc)

        pps_matched_micros.append(pps_micros)
        pps_matched_utc.append(float(utc_second))

    # Perform linear regression
    # To avoid numerical inaccuracies, subtract the minimum micros value
    # Use global minimum if provided, otherwise use minimum from PPS data
    if global_min_micros is None:
        min_micros = min(pps_matched_micros)
    else:
        min_micros = global_min_micros

    pps_matched_micros_offset = [m - min_micros for m in pps_matched_micros]

    slope, intercept_offset, r_value, p_value, std_err = stats.linregress(
        pps_matched_micros_offset, pps_matched_utc
    )

    # Adjust intercept to account for the offset we subtracted
    intercept = intercept_offset - slope * min_micros

    logger.info(f"PPS regression: slope={slope:.12f}, intercept={intercept:.6f}")
    logger.info(f"  R²={r_value**2:.9f}, p-value={p_value:.2e}, std_err={std_err:.2e}")
    logger.info(f"  Used {len(pps_matched_micros)} PPS-GNSS matched pairs")

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
    Uses the unwrapped micros_reading values.

    Args:
        pps_list: List of PPS fixes
        gnss_list: List of GNSS readings
        imu_list: List of IMU readings
        slope: Regression slope
        intercept: Regression intercept
    """
    # Apply to PPS using unwrapped values
    if pps_list:
        for pps in pps_list:
            micros_unwrapped = pps.micros_reading_unwrapped
            if micros_unwrapped is None:
                micros_unwrapped = pps.micros_reading
            pps.utc_timestamp_from_pps_regression = (
                slope * micros_unwrapped + intercept
            )
            pps.datetime_timestamp_from_pps_regression = datetime.fromtimestamp(
                pps.utc_timestamp_from_pps_regression, tz=timezone.utc
            )

    # Apply to GNSS using unwrapped values
    if gnss_list:
        for gnss in gnss_list:
            micros_unwrapped = gnss.micros_reading_unwrapped
            if micros_unwrapped is None:
                micros_unwrapped = gnss.micros_reading
            gnss.utc_timestamp_from_pps_regression = (
                slope * micros_unwrapped + intercept
            )
            gnss.datetime_timestamp_from_pps_regression = datetime.fromtimestamp(
                gnss.utc_timestamp_from_pps_regression, tz=timezone.utc
            )

    # Apply to IMU using unwrapped values
    if imu_list:
        for imu in imu_list:
            micros_unwrapped = imu.micros_reading_unwrapped
            if micros_unwrapped is None:
                micros_unwrapped = imu.micros_reading
            imu.utc_timestamp_from_pps_regression = (
                slope * micros_unwrapped + intercept
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
    micros_reading = struct.unpack("<I", data[:4])[0]
    return PPSFix(micros_reading=micros_reading)


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

    micros_reading = values[0]
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
        micros_reading=micros_reading,
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
    values = struct.unpack("<IHhhhhhh", data[:18])
    micros_reading = values[0]
    counter = values[1]
    acc_x, acc_y, acc_z = values[2], values[3], values[4]
    gyr_x, gyr_y, gyr_z = values[5], values[6], values[7]

    acc_x_mg = acc_x * acc_sensitivity
    acc_y_mg = acc_y * acc_sensitivity
    acc_z_mg = acc_z * acc_sensitivity
    gyr_x_mdps = gyr_x * gyr_sensitivity
    gyr_y_mdps = gyr_y * gyr_sensitivity
    gyr_z_mdps = gyr_z * gyr_sensitivity

    return IMUReading(
        micros_reading=micros_reading,
        counter=counter,
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


def compute_pps_mismatch_statistics(
    pps_list: list[PPSFix],
    show_plot: bool = False
) -> None:
    """Compute and display PPS mismatch statistics and plot.
    
    For each PPS entry, computes the mismatch between the UTC datetime 
    from linear regression and the closest UTC second. Displays statistics
    and an ASCII terminal plot showing how the mismatch varies over time.
    
    Args:
        pps_list: List of PPS fixes with regression timestamps computed
        show_plot: If True, display ASCII plot of mismatch vs time
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

    if show_plot:
        if not GNUPLOT_AVAILABLE:
            logger.error(
                "Cannot display plots: gnuplotlib is not available. "
                "Install with: pip install gnuplotlib"
            )
            return
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
    unwrap_stats: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print summary statistics about the parsed data.

    Args:
        pps_list: List of parsed PPS entries
        gnss_list: List of parsed GNSS entries
        imu_list: List of parsed IMU entries
        unwrap_stats: Optional dictionary with unwrap/jump statistics
    """
    logger.info("=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)

    logger.info("Number of messages parsed:")
    logger.info(f"  PPS:  {len(pps_list):6d}")
    logger.info(f"  GNSS: {len(gnss_list):6d}")
    logger.info(f"  IMU:  {len(imu_list):6d}")

    # Print unwrap statistics if available
    if unwrap_stats:
        logger.info("")
        logger.info("Unwrap and jump statistics:")
        for data_type, stats in unwrap_stats.items():
            logger.info(f"  {data_type}:")
            for field, counts in stats.items():
                logger.info(f"    {field}:")
                logger.info(f"      Wraps: {counts['wraps']}")
                logger.info(f"      Jumps: {counts['jumps']}")

    if len(imu_list) >= 2:
        # Find first and last non-jumped entries for duration calculation
        first_idx = 0
        last_idx = len(imu_list) - 1

        # Get jump indices if available
        jump_indices = set()
        if unwrap_stats and "IMU" in unwrap_stats:
            if "micros_reading" in unwrap_stats["IMU"]:
                jumps = unwrap_stats["IMU"]["micros_reading"].get("jump_indices")
                if jumps is not None:
                    jump_indices = set(jumps)

        # Walk backwards from end to find last non-jumped entry
        while last_idx > first_idx and last_idx in jump_indices:
            last_idx -= 1

        # Walk forward from start to find first non-jumped entry
        while first_idx < last_idx and first_idx in jump_indices:
            first_idx += 1

        if first_idx < last_idx:
            # Use unwrapped micros if available, otherwise use raw
            if imu_list[first_idx].micros_reading_unwrapped is not None:
                first_micros = imu_list[first_idx].micros_reading_unwrapped
                last_micros = imu_list[last_idx].micros_reading_unwrapped
            else:
                first_micros = imu_list[first_idx].micros_reading
                last_micros = imu_list[last_idx].micros_reading

            duration_us = last_micros - first_micros
            duration_s = duration_us / 1e6
            duration_min = duration_s / 60.0

            logger.info("")
            logger.info("File duration (from IMU timestamps, excluding jumps):")
            logger.info(f"  First micros: {first_micros}")
            logger.info(f"  Last micros:  {last_micros}")
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
            logger.warning("All IMU entries have jumps, cannot compute duration")
    else:
        logger.warning("Not enough IMU data to compute duration")

    logger.info("=" * 60)


def decode_file(
    input_file: Path,
    output_dir: Path | None = None,
    show_plots: bool = False,
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
        show_plots: If True, display ASCII plots (e.g., PPS mismatch plot)
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

            # Skip padding bytes (C struct alignment adds 2 bytes)
            idx += IMU_PADDING

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

    # Unwrap and detect jumps for all relevant fields
    logger.info("Unwrapping and detecting jumps in data...")
    unwrap_stats = {}

    # Process PPS micros_reading
    if pps_list:
        pps_micros = np.array([p.micros_reading for p in pps_list])
        pps_micros_unwrapped, pps_micros_wraps, pps_micros_jumps = unwrap_array(
            pps_micros, max_value=2**32
        )
        for i, pps in enumerate(pps_list):
            pps.micros_reading_unwrapped = int(pps_micros_unwrapped[i])

        unwrap_stats["PPS"] = {
            "micros_reading": {
                "wraps": 0 if pps_micros_wraps is None else len(pps_micros_wraps),
                "jumps": 0 if pps_micros_jumps is None else len(pps_micros_jumps),
                "wrap_indices": pps_micros_wraps,
                "jump_indices": pps_micros_jumps,
            }
        }

    # Process GNSS micros_reading
    if gnss_list:
        gnss_micros = np.array([g.micros_reading for g in gnss_list])
        gnss_micros_unwrapped, gnss_micros_wraps, gnss_micros_jumps = unwrap_array(
            gnss_micros, max_value=2**32
        )
        for i, gnss in enumerate(gnss_list):
            gnss.micros_reading_unwrapped = int(gnss_micros_unwrapped[i])

        unwrap_stats["GNSS"] = {
            "micros_reading": {
                "wraps": 0 if gnss_micros_wraps is None else len(gnss_micros_wraps),
                "jumps": 0 if gnss_micros_jumps is None else len(gnss_micros_jumps),
                "wrap_indices": gnss_micros_wraps,
                "jump_indices": gnss_micros_jumps,
            }
        }

    # Process IMU micros_reading and counter
    if imu_list:
        imu_micros = np.array([i.micros_reading for i in imu_list])
        imu_micros_unwrapped, imu_micros_wraps, imu_micros_jumps = unwrap_array(
            imu_micros, max_value=2**32
        )

        imu_counter = np.array([i.counter for i in imu_list])
        imu_counter_unwrapped, imu_counter_wraps, imu_counter_jumps = unwrap_array(
            imu_counter, max_value=2**16, jump_threshold=1
        )

        for i, imu in enumerate(imu_list):
            imu.micros_reading_unwrapped = int(imu_micros_unwrapped[i])
            imu.counter_unwrapped = int(imu_counter_unwrapped[i])

        unwrap_stats["IMU"] = {
            "micros_reading": {
                "wraps": 0 if imu_micros_wraps is None else len(imu_micros_wraps),
                "jumps": 0 if imu_micros_jumps is None else len(imu_micros_jumps),
                "wrap_indices": imu_micros_wraps,
                "jump_indices": imu_micros_jumps,
            },
            "counter": {
                "wraps": 0 if imu_counter_wraps is None else len(imu_counter_wraps),
                "jumps": 0 if imu_counter_jumps is None else len(imu_counter_jumps),
                "wrap_indices": imu_counter_wraps,
                "jump_indices": imu_counter_jumps,
            },
        }

    # Print summary statistics
    print_summary_statistics(pps_list, gnss_list, imu_list, unwrap_stats)

    # Compute and apply PPS regression
    # Find global minimum micros across all data types
    all_micros = []
    if pps_list:
        all_micros.extend([p.micros_reading for p in pps_list])
    if gnss_list:
        all_micros.extend([g.micros_reading for g in gnss_list])
    if imu_list:
        all_micros.extend([i.micros_reading for i in imu_list])

    global_min_micros = min(all_micros) if all_micros else 0
    logger.info(f"Global minimum micros reading: {global_min_micros}")

    logger.info("Computing PPS to UTC timestamp regression...")
    regression = compute_pps_regression(pps_list, gnss_list, global_min_micros)
    if regression is None:
        logger.warning("Skipping PPS regression due to insufficient data")
    else:
        slope, intercept = regression
        apply_pps_regression(pps_list, gnss_list, imu_list, slope, intercept)

        # Compute and display PPS mismatch statistics
        compute_pps_mismatch_statistics(pps_list, show_plot=show_plots)

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

    # Add unwrap statistics to output
    output_files["unwrap_stats"] = unwrap_stats

    return output_files
