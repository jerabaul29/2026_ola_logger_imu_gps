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
    # Outlier detection flags
    latitude_dd_stdchecked: bool = False
    longitude_dd_stdchecked: bool = False
    ned_vel_north_mmps_stdchecked: bool = False
    ned_vel_east_mmps_stdchecked: bool = False
    ned_vel_down_mmps_stdchecked: bool = False


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
    # Outlier detection flags
    acc_x_mg_stdchecked: bool = False
    acc_y_mg_stdchecked: bool = False
    acc_z_mg_stdchecked: bool = False
    gyr_x_mdps_stdchecked: bool = False
    gyr_y_mdps_stdchecked: bool = False
    gyr_z_mdps_stdchecked: bool = False


def parse_header(
    file_path: Path,
    markers: tuple[bytes, bytes, bytes] = (PPS_MARKER, GPS_MARKER, IMU_MARKER),
    search_bytes: int = HEADER_SEARCH_BYTES,
) -> tuple[dict[str, Any], str]:
    """Parse the header of the data file and extract metadata.

    Args:
        file_path: Path to the data file
        markers: Tuple of markers that indicate start of data section
        search_bytes: Number of bytes to scan from start of file for header

    Returns:
        Tuple of (header_info dict, header_text string)
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
    return header_info, header_text


def unwrap_array(
    values: np.ndarray,
    max_value: int,
    wrap_threshold: float | None = None,
    jump_threshold: float | None = None,
    initial_offset: int = 0,
    prev_raw_value: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int | None]:
    """Unwrap potentially wrapping array and detect anomalous jumps.

    Handles overflow in fixed-width integer timestamps (uint32_t micros, uint16_t counters)
    by detecting wrap-around events and applying offset corrections. Also identifies
    anomalous jumps that indicate missed data or timing glitches.

    Algorithm:
    1. Scan array for large negative jumps (diff < -wrap_threshold) → wrap detected
    2. Apply cumulative offset (add max_value) to all values after each wrap
    3. On unwrapped data, detect anomalous jumps:
       - Any negative jump (should be monotonic after unwrapping)
       - Any positive jump > jump_threshold (unexpectedly large time gap)

    Args:
        values: Array of potentially wrapping values (e.g., micros_reading, counter)
        max_value: Maximum value before wrapping (e.g., 2**32 for uint32, 2**16 for uint16)
        wrap_threshold: Threshold for wrap detection as fraction of max_value
                       (default: 0.75 * max_value, meaning negative jumps > 75% are wraps)
        jump_threshold: Threshold for anomalous jump detection
                       (default: 0.1 * max_value for timestamps, 1 for counters)
        initial_offset: Initial unwrap offset from previous segment (default: 0)
        prev_raw_value: Last raw value from previous segment for wrap detection at boundary

    Returns:
        Tuple of:
        - unwrapped_array: Array with wrapping corrected (int64 to avoid overflow)
        - wrap_indices: Indices where wraps occurred (None if no wraps detected)
        - jump_indices: Indices where anomalous jumps occurred (None if no jumps detected)
        - final_offset: Final unwrap offset to pass to next segment
        - last_raw_value: Last raw value to pass to next segment

    Example:
        >>> values = np.array([2**32-1000, 2**32-500, 100])  # Wraps at index 2
        >>> unwrapped, wraps, jumps, final_offset, last_raw = unwrap_array(values, max_value=2**32)
        >>> unwrapped
        array([4294966296, 4294966796, 4294967396])  # Monotonic after unwrapping
        >>> wraps
        array([2])  # Wrap detected at index 2
    """
    if len(values) == 0:
        return np.array([]), None, None, initial_offset, prev_raw_value

    if wrap_threshold is None:
        wrap_threshold = 0.75 * max_value
    if jump_threshold is None:
        jump_threshold = 0.1 * max_value

    # Step 1: Detect wraps and unwrap
    unwrapped = np.zeros_like(values, dtype=np.int64)
    offset = initial_offset
    wrap_indices_list = []

    # Check for wrap at segment boundary (first value vs previous segment's last value)
    if prev_raw_value is not None:
        diff = values[0] - prev_raw_value
        if diff < 0 and abs(diff) > wrap_threshold:
            offset += max_value
            wrap_indices_list.append(0)
    
    unwrapped[0] = values[0] + offset

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
    
    # Return last raw value for next segment
    last_raw_value = int(values[-1]) if len(values) > 0 else None

    return unwrapped, wrap_indices, jump_indices, offset, last_raw_value


def detect_outliers_stdcheck(
    values: np.ndarray,
    n_neighbors: int = 6,
    n_sigma: float = 5.0
) -> np.ndarray:
    """Detect outliers in a time series using neighboring values statistics.
    
    For each point in the time series, computes statistics from its N nearest
    neighbors and flags points that deviate by more than n_sigma standard
    deviations from the local mean.
    
    Algorithm:
    1. For each index i, find N closest neighboring indices
       - Interior points: symmetric neighbors (e.g., i-3, i-2, i-1, i+1, i+2, i+3 for N=6)
       - Edge points: asymmetric neighbors (e.g., i-1, i+1, i+2, i+3, i+4, i+5)
    2. Compute mean and std from these neighbors (excluding point i itself)
    3. Flag point i if |value[i] - mean| > n_sigma * std
    
    Special cases:
    - NaN/inf values are flagged as outliers
    - When all neighbors are identical (std=0), flag if value differs
    - Empty arrays or arrays with all NaN return empty result
    
    Args:
        values: 1D array of time series values
        n_neighbors: Number of neighbors to use for statistics (default: 6)
        n_sigma: Number of standard deviations for outlier threshold (default: 5.0)
        
    Returns:
        Array of indices where outliers were detected (empty if none found)
        
    Example:
        >>> data = np.array([1.0, 1.1, 1.0, 10.0, 0.9, 1.1, 1.0])
        >>> outliers = detect_outliers_stdcheck(data, n_neighbors=6, n_sigma=5.0)
        >>> outliers
        array([3])  # Index 3 (value=10.0) is an outlier
    """
    # Input validation
    if len(values) == 0:
        return np.array([], dtype=np.int64)
    
    # Check for all-NaN or all-inf arrays
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        # All values are NaN/inf - return all indices as outliers
        return np.arange(len(values), dtype=np.int64)
    
    if len(values) < n_neighbors + 1:
        # Not enough data points for meaningful outlier detection
        # Flag NaN/inf values only
        return np.where(~finite_mask)[0].astype(np.int64)
    
    outlier_indices = []
    n = len(values)
    
    # Number of neighbors on each side (for symmetric case)
    half_neighbors = n_neighbors // 2
    
    for i in range(n):
        # Flag NaN/inf values immediately
        if not np.isfinite(values[i]):
            outlier_indices.append(i)
            continue
        
        # Determine neighbor indices based on position
        if i < half_neighbors:
            # Near start: take neighbors to the right
            neighbor_start = 0
            neighbor_end = min(n_neighbors + 1, n)
        elif i >= n - half_neighbors:
            # Near end: take neighbors to the left
            neighbor_start = max(0, n - n_neighbors - 1)
            neighbor_end = n
        else:
            # Interior: symmetric neighbors
            neighbor_start = i - half_neighbors
            neighbor_end = i + half_neighbors + 1
        
        # Get neighbor values (excluding the point itself)
        neighbor_indices = list(range(neighbor_start, neighbor_end))
        if i in neighbor_indices:
            neighbor_indices.remove(i)
        
        # Ensure we have exactly n_neighbors (or as many as possible)
        neighbor_indices = neighbor_indices[:n_neighbors]
        
        if len(neighbor_indices) < 2:
            # Need at least 2 neighbors to compute std
            continue
            
        neighbor_values = values[neighbor_indices]
        
        # Filter out NaN/inf from neighbors
        finite_neighbors = neighbor_values[np.isfinite(neighbor_values)]
        if len(finite_neighbors) < 2:
            # Not enough valid neighbors for statistics
            continue
        
        # Compute statistics from valid neighbors
        mean_val = np.mean(finite_neighbors)
        std_val = np.std(finite_neighbors, ddof=1)  # Use sample std
        
        # Check if current value is an outlier
        deviation = abs(values[i] - mean_val)
        if std_val > 0:
            # Normal case: check if deviation exceeds threshold
            if deviation > n_sigma * std_val:
                outlier_indices.append(i)
        else:
            # When std is 0 (all neighbors identical), flag if value differs from mean
            if deviation > 0:
                outlier_indices.append(i)
    
    return np.array(outlier_indices, dtype=np.int64)


def apply_outlier_flags(
    data_list: list,
    field_name: str,
    outlier_indices: np.ndarray
) -> None:
    """Apply outlier flags to a list of dataclass objects.
    
    This is a helper function to efficiently set boolean flags for detected outliers.
    
    Args:
        data_list: List of dataclass objects (IMUReading or GNSSReading)
        field_name: Name of the boolean flag field to set (e.g., 'acc_x_mg_stdchecked')
        outlier_indices: Array of indices where outliers were detected
        
    Example:
        >>> imu_list = [IMUReading(...), IMUReading(...), ...]
        >>> outliers = detect_outliers_stdcheck(acc_x_values)
        >>> apply_outlier_flags(imu_list, 'acc_x_mg_stdchecked', outliers)
    """
    for idx in outlier_indices:
        setattr(data_list[idx], field_name, True)


def compute_pps_regression(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    global_min_micros: int | None = None,
) -> tuple[float, float] | None:
    """Compute linear regression from PPS micros to UTC timestamps.

    This function synchronizes MCU microsecond timestamps to absolute UTC time
    by establishing a linear mapping between PPS events and GNSS-provided UTC
    timestamps. The regression allows sub-millisecond accuracy for all sensor
    data timestamps.

    Process:
    1. Uses unwrapped micros timestamps for both PPS and GNSS data
    2. For each PPS event, finds the temporally closest GNSS measurement
    3. Uses GNSS UTC time to determine which second boundary the PPS marks
    4. Performs linear regression with improved normalization:
       - Subtracts minimum from both micros and UTC timestamps
       - Converts micros offset to seconds
       - Normalizes both quantities to max value of 1.0
       - This improves numerical stability and precision of the regression
    5. Transforms coefficients back to original scale: UTC_time = slope × micros + intercept

    Args:
        pps_list: List of PPS fixes (must have micros_reading_unwrapped populated)
        gnss_list: List of GNSS readings (must have micros_reading_unwrapped populated)
        global_min_micros: Minimum micros value across all data types for numerical
                          stability (defaults to min of pps_list if not provided)

    Returns:
        Tuple of (slope, intercept) for the linear regression: UTC = slope*micros + intercept
        Returns None if insufficient data (empty lists or fewer than 2 PPS entries)
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
    pps_micros_unwrapped = np.array([
        p.micros_reading_unwrapped if p.micros_reading_unwrapped is not None
        else p.micros_reading
        for p in pps_list
    ], dtype=np.int64)
    gnss_micros_unwrapped = np.array([
        g.micros_reading_unwrapped if g.micros_reading_unwrapped is not None
        else g.micros_reading
        for g in gnss_list
    ], dtype=np.int64)

    # For each PPS entry, find the closest GNSS entry by micros
    # and determine which UTC second boundary the PPS marks
    pps_matched_micros = []
    pps_matched_utc = []

    # Use binary search for efficient closest neighbor finding
    # Sort GNSS micros if not already sorted (should be in chronological order)
    gnss_sorted_indices = np.argsort(gnss_micros_unwrapped)
    gnss_micros_sorted = gnss_micros_unwrapped[gnss_sorted_indices]

    for pps_micros in pps_micros_unwrapped:
        # Find insertion point using binary search
        insert_idx = np.searchsorted(gnss_micros_sorted, pps_micros)
        
        # Check neighbors around insertion point to find closest
        candidates = []
        if insert_idx > 0:
            candidates.append(insert_idx - 1)
        if insert_idx < len(gnss_micros_sorted):
            candidates.append(insert_idx)
        
        # Find the closest candidate
        if not candidates:
            continue
            
        closest_sorted_idx = min(
            candidates,
            key=lambda idx: abs(gnss_micros_sorted[idx] - pps_micros)
        )
        
        # Map back to original GNSS list index
        closest_gnss_idx = gnss_sorted_indices[closest_sorted_idx]

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

    # Perform linear regression with improved normalization
    # To avoid numerical inaccuracies:
    # 1. Subtract minimum from both micros and UTC
    # 2. Convert micros offset to seconds
    # 3. Normalize both to have max value of 1.0
    
    # Use global minimum if provided, otherwise use minimum from PPS data
    if global_min_micros is None:
        min_micros = min(pps_matched_micros)
    else:
        min_micros = global_min_micros
    
    min_utc = min(pps_matched_utc)
    
    # Subtract minimums
    pps_matched_micros_offset = [m - min_micros for m in pps_matched_micros]
    pps_matched_utc_offset = [u - min_utc for u in pps_matched_utc]
    
    # Convert micros to seconds
    pps_matched_micros_offset_sec = [m / 1e6 for m in pps_matched_micros_offset]
    
    # Normalize both to max value of 1.0
    max_micros_sec = max(pps_matched_micros_offset_sec)
    max_utc = max(pps_matched_utc_offset)
    
    # Avoid division by zero (shouldn't happen with valid data)
    if max_micros_sec == 0 or max_utc == 0:
        logger.error("Cannot normalize: max value is zero")
        return None
    
    pps_matched_micros_normalized = [m / max_micros_sec for m in pps_matched_micros_offset_sec]
    pps_matched_utc_normalized = [u / max_utc for u in pps_matched_utc_offset]
    
    # Perform linear regression on normalized data
    slope_norm, intercept_norm, r_value, p_value, std_err = stats.linregress(
        pps_matched_micros_normalized, pps_matched_utc_normalized
    )
    
    # Transform back to original scale
    # y_norm = slope_norm * x_norm + intercept_norm
    # (y - min_utc) / max_utc = slope_norm * ((x - min_micros)/1e6) / max_micros_sec + intercept_norm
    # y = slope_norm * max_utc * (x - min_micros) / (1e6 * max_micros_sec) + intercept_norm * max_utc + min_utc
    # y = slope_final * x + intercept_final
    # where slope_final = slope_norm * max_utc / (1e6 * max_micros_sec)
    #       intercept_final = -slope_final * min_micros + intercept_norm * max_utc + min_utc
    
    slope = slope_norm * max_utc / (1e6 * max_micros_sec)
    intercept = -slope * min_micros + intercept_norm * max_utc + min_utc

    logger.info(f"PPS regression: slope={slope:.12f}, intercept={intercept:.6f}")
    logger.info(f"  R²={r_value**2:.9f}, p-value={p_value:.2e}, std_err={std_err:.2e}")
    logger.info(f"  Used {len(pps_matched_micros)} PPS-GNSS matched pairs")
    logger.info(f"  Normalization: micros range {min_micros} to {min_micros + max_micros_sec*1e6:.0f} µs")
    logger.info(f"  Normalization: UTC range {min_utc:.1f} to {min_utc + max_utc:.1f} s")

    return (slope, intercept)


def apply_pps_regression(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    imu_list: list[IMUReading],
    slope: float,
    intercept: float,
) -> None:
    """Apply PPS regression to all data entries for synchronized UTC timestamps.

    Modifies dataclass objects in-place, adding UTC timestamp fields computed
    from the linear regression: UTC = slope × micros_unwrapped + intercept

    This provides absolute UTC timestamps (both as POSIX floats and timezone-aware
    datetime objects) for all sensor measurements, enabling precise time
    synchronization across PPS, GNSS, and IMU data streams.

    Args:
        pps_list: List of PPS fixes to update
        gnss_list: List of GNSS readings to update
        imu_list: List of IMU readings to update
        slope: Regression slope (microseconds to seconds conversion factor)
        intercept: Regression intercept (seconds)

    Note:
        Uses unwrapped micros_reading values to handle uint32_t overflow correctly.
        Falls back to raw micros_reading if unwrapped value is None.
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
    """Compute and display PPS mismatch statistics to assess regression quality.
    
    Evaluates how well the linear regression aligns PPS events to exact UTC
    second boundaries. Each PPS pulse should occur at the start of a UTC second
    (e.g., 12:34:56.000). This function computes the deviation between the
    regression-predicted timestamp and the nearest second boundary.
    
    Displays:
    - Maximum absolute mismatch (ms)
    - Mean mismatch (ms) - should be near zero for unbiased regression
    - RMS mismatch (ms) - overall accuracy metric
    - Optional ASCII plot of mismatch vs time (requires gnuplotlib)
    
    Args:
        pps_list: List of PPS fixes with utc_timestamp_from_pps_regression populated
        show_plot: If True, display ASCII terminal plot using gnuplotlib
                  (silently skips if gnuplotlib not available)
    
    Note:
        Typical good results: max < 5ms, RMS < 2ms for R² > 0.999999
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

    Displays:
    - Number of messages parsed for each data type
    - Unwrap and jump statistics (wraps and anomalous jumps detected)
    - File duration based on IMU timestamps (excluding anomalous entries)
    - Effective sampling rates for each data type

    Args:
        pps_list: List of parsed PPS entries
        gnss_list: List of parsed GNSS entries
        imu_list: List of parsed IMU entries
        unwrap_stats: Optional dictionary with unwrap/jump statistics containing
                     wrap and jump counts for each field of each data type
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
        for data_type, field_stats in unwrap_stats.items():
            logger.info(f"  {data_type}:")
            for field, counts in field_stats.items():
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
            
            # Handle counter wraps (if duration is negative, assume wrap occurred)
            if duration_us < 0:
                duration_us += 2**32  # uint32_t wrap
            
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


CORRUPTION_SCAN_BYTES = 1024


def scan_for_next_valid_marker(
    content: bytes,
    start_idx: int,
    valid_markers: tuple[bytes, ...],
    scan_bytes: int = CORRUPTION_SCAN_BYTES,
) -> int | None:
    """Scan ahead in content for the next valid data entry marker.

    Used for corruption recovery: when an unexpected byte is encountered,
    this function searches forward to find the next valid entry marker
    to resume parsing.

    Args:
        content: Full file content as bytes
        start_idx: Position to start scanning from
        valid_markers: Tuple of valid marker bytes to search for (e.g., PPS, GPS, IMU markers)
        scan_bytes: Maximum number of bytes to scan (default: 1024)

    Returns:
        Index of the next valid marker if found, None otherwise
    """
    end_idx = min(start_idx + scan_bytes, len(content))
    scan_region = content[start_idx:end_idx]

    # Find earliest occurrence of any valid marker
    earliest_pos = None
    earliest_offset = float('inf')

    for marker in valid_markers:
        pos = scan_region.find(marker)
        if pos != -1 and pos < earliest_offset:
            earliest_offset = pos
            earliest_pos = start_idx + pos

    return earliest_pos


def handle_junk_bytes(
    content: bytes,
    idx: int,
    next_byte: int,
    junk_start: int,
    entry_type: str,
    markers: tuple[bytes, ...],
    footer_marker: bytes,
    pps_list: list,
    gnss_list: list,
    imu_list: list,
) -> tuple[int, bool]:
    """Handle junk bytes after an entry.
    
    Args:
        content: Full file content
        idx: Current index after entry
        next_byte: The unexpected byte found
        junk_start: Offset where junk started
        entry_type: Type of entry ("PPS", "GNSS", "IMU")
        markers: Valid entry markers for recovery
        footer_marker: Footer marker bytes
        pps_list, gnss_list, imu_list: Current parsed entries
        
    Returns:
        Tuple of (new_idx, should_break)
        - new_idx: Updated index position
        - should_break: Whether to break from parsing loop
    """
    junk_bytes = []
    
    # Skip junk bytes until we find a valid marker
    while idx < len(content):
        b = content[idx]
        if b == ord(b'\n') or footer_marker in content[idx:idx+20]:
            # Found valid marker, stop skipping
            break
        junk_bytes.append(b)
        idx += 1
        
        # Safety: don't skip more than a reasonable amount
        if len(junk_bytes) >= CORRUPTION_SCAN_BYTES:
            # Too much junk - treat as serious corruption
            logger.warning(
                f"Unexpected byte 0x{next_byte:02x} at offset {junk_start} after {entry_type} entry "
                f"(>{CORRUPTION_SCAN_BYTES} junk bytes)"
            )
            logger.error("Scanning ahead for valid marker...")
            
            # Scan ahead for next valid entry
            next_marker_idx = scan_for_next_valid_marker(content, idx, markers)
            
            if next_marker_idx is not None:
                bytes_skipped = next_marker_idx - junk_start
                logger.info(
                    f"Recovered at offset {next_marker_idx} ({bytes_skipped} bytes skipped)"
                )
                return next_marker_idx, False
            else:
                logger.error(
                    f"Recovery failed. Parsed {len(pps_list)} PPS, "
                    f"{len(gnss_list)} GNSS, {len(imu_list)} IMU before corruption"
                )
                return idx, True
    
    # Successfully skipped small amount of junk
    if 0 < len(junk_bytes) < CORRUPTION_SCAN_BYTES:
        logger.warning(
            f"Skipped {len(junk_bytes)} junk byte(s) at offset {junk_start} after {entry_type} "
            f"(first: 0x{junk_bytes[0]:02x})"
        )
    
    return idx, False


def process_pps_entry(
    content: bytes,
    idx: int,
    pps_list: list,
    gnss_list: list,
    imu_list: list,
    pps_marker: bytes,
    gps_marker: bytes,
    imu_marker: bytes,
    footer_marker: bytes,
    pps_struct_size: int,
) -> tuple[int, bool]:
    """Process a single PPS entry.
    
    Returns:
        Tuple of (new_idx, should_break)
    """
    # Check we have enough bytes
    line_end = idx + PPS_LINE_SIZE
    if line_end > len(content):
        logger.warning(
            f"Incomplete PPS entry at offset {idx}: "
            f"need {PPS_LINE_SIZE} bytes, only {len(content) - idx} available"
        )
        logger.error(
            f"File truncated. Parsed {len(pps_list)} PPS, "
            f"{len(gnss_list)} GNSS, {len(imu_list)} IMU entries before truncation"
        )
        return idx, True
    
    # Parse entry
    idx += 4
    pps_data = content[idx : idx + pps_struct_size]
    try:
        pps_entry = parse_pps_entry(pps_data)
        pps_list.append(pps_entry)
    except (struct.error, AssertionError) as e:
        logger.warning(f"Failed to parse PPS entry at offset {idx}: {e}")
        logger.error(f"Parsing aborted (data length={len(pps_data)}, expected={pps_struct_size})")
        raise
    idx += pps_struct_size
    
    # Check next byte is valid
    if idx < len(content):
        next_byte = content[idx]
        if next_byte == ord(b'\n') or footer_marker in content[idx:idx+20]:
            return idx, False
        else:
            # Handle junk bytes
            new_idx, should_break = handle_junk_bytes(
                content, idx, next_byte, idx, "PPS",
                (pps_marker, gps_marker, imu_marker, footer_marker),
                footer_marker, pps_list, gnss_list, imu_list
            )
            return new_idx, should_break
    
    return idx, False


def process_gnss_entry(
    content: bytes,
    idx: int,
    pps_list: list,
    gnss_list: list,
    imu_list: list,
    pps_marker: bytes,
    gps_marker: bytes,
    imu_marker: bytes,
    footer_marker: bytes,
    gps_struct_size: int,
) -> tuple[int, bool]:
    """Process a single GNSS entry.
    
    Returns:
        Tuple of (new_idx, should_break)
    """
    # Check we have enough bytes
    line_end = idx + GPS_LINE_SIZE
    if line_end > len(content):
        logger.warning(
            f"Incomplete GNSS entry at offset {idx}: "
            f"need {GPS_LINE_SIZE} bytes, only {len(content) - idx} available"
        )
        logger.error(
            f"File truncated. Parsed {len(pps_list)} PPS, "
            f"{len(gnss_list)} GNSS, {len(imu_list)} IMU entries before truncation"
        )
        return idx, True
    
    # Parse entry
    idx += 4
    gnss_data = content[idx : idx + gps_struct_size]
    try:
        gnss_entry = parse_gnss_entry(gnss_data)
        gnss_list.append(gnss_entry)
    except (struct.error, AssertionError) as e:
        logger.warning(f"Failed to parse GNSS entry at offset {idx}: {e}")
        logger.error(f"Parsing aborted (data length={len(gnss_data)}, expected={gps_struct_size})")
        raise
    idx += gps_struct_size
    
    # Check next byte is valid
    if idx < len(content):
        next_byte = content[idx]
        if next_byte == ord(b'\n') or footer_marker in content[idx:idx+20]:
            return idx, False
        else:
            # Handle junk bytes
            new_idx, should_break = handle_junk_bytes(
                content, idx, next_byte, idx, "GNSS",
                (pps_marker, gps_marker, imu_marker, footer_marker),
                footer_marker, pps_list, gnss_list, imu_list
            )
            return new_idx, should_break
    
    return idx, False


def process_imu_entry(
    content: bytes,
    idx: int,
    pps_list: list,
    gnss_list: list,
    imu_list: list,
    pps_marker: bytes,
    gps_marker: bytes,
    imu_marker: bytes,
    footer_marker: bytes,
    imu_struct_size: int,
    acc_sensitivity: float,
    gyr_sensitivity: float,
    prev_imu_micros: int | None = None,
) -> tuple[int, bool, bool]:
    """Process a single IMU entry.
    
    Args:
        prev_imu_micros: Previous IMU micros value for jump detection (None if first in segment)
    
    Returns:
        Tuple of (new_idx, should_break, jump_detected)
        - jump_detected: True if a micros jump was detected that should trigger segmentation
    """
    # Check we have enough bytes
    line_end = idx + IMU_LINE_SIZE
    if line_end > len(content):
        logger.warning(
            f"Incomplete IMU entry at offset {idx}: "
            f"need {IMU_LINE_SIZE} bytes, only {len(content) - idx} available"
        )
        logger.error(
            f"File truncated. Parsed {len(pps_list)} PPS, "
            f"{len(gnss_list)} GNSS, {len(imu_list)} IMU entries before truncation"
        )
        return idx, True, False
    
    # Parse entry
    idx += 4
    imu_data = content[idx : idx + imu_struct_size]
    try:
        imu_entry = parse_imu_entry(imu_data, acc_sensitivity, gyr_sensitivity)
        
        # Check for micros jump if we have previous value
        jump_detected = False
        if prev_imu_micros is not None:
            micros_diff = imu_entry.micros_reading - prev_imu_micros
            
            # Check for negative jump (backwards in time)
            if micros_diff < 0:
                logger.warning(
                    f"IMU micros negative jump detected at offset {idx}: "
                    f"{prev_imu_micros} → {imu_entry.micros_reading} "
                    f"(diff={micros_diff} µs). Starting new segment."
                )
                jump_detected = True
            # Check for large positive jump (> 1 second = 1,000,000 µs)
            elif micros_diff > 1_000_000:
                logger.warning(
                    f"IMU micros large jump detected at offset {idx}: "
                    f"{prev_imu_micros} → {imu_entry.micros_reading} "
                    f"(diff={micros_diff} µs = {micros_diff/1e6:.3f}s). Starting new segment."
                )
                jump_detected = True
        
        imu_list.append(imu_entry)
    except (struct.error, AssertionError) as e:
        logger.warning(f"Failed to parse IMU entry at offset {idx}: {e}")
        logger.error(f"Parsing aborted (data length={len(imu_data)}, expected={imu_struct_size})")
        raise
    idx += imu_struct_size
    
    # Skip padding bytes
    idx += IMU_PADDING
    
    # Check next byte is valid
    if idx < len(content):
        next_byte = content[idx]
        if next_byte == ord(b'\n') or footer_marker in content[idx:idx+20]:
            return idx, False, jump_detected
        else:
            # Handle junk bytes
            new_idx, should_break = handle_junk_bytes(
                content, idx, next_byte, idx, "IMU",
                (pps_marker, gps_marker, imu_marker, footer_marker),
                footer_marker, pps_list, gnss_list, imu_list
            )
            return new_idx, should_break, jump_detected
    
    return idx, False, jump_detected


def parse_binary_content(
    content: bytes,
    header_info: dict,
    pps_marker: bytes,
    gps_marker: bytes,
    imu_marker: bytes,
    footer_marker: bytes,
) -> list[dict[str, list]]:
    """Parse binary content and extract all PPS, GNSS, and IMU entries in segments.
    
    Segments are created based on two conditions:
    1. Time-based: once n_imus_per_segment IMU entries are reached (~1 minute)
    2. Jump-based: when IMU micros has a negative jump or positive jump > 1 second
    
    Args:
        content: Full file content as bytes
        header_info: Parsed header information
        pps_marker, gps_marker, imu_marker, footer_marker: Entry markers
        
    Returns:
        List of segment dicts, each containing {'pps_list': [], 'gnss_list': [], 'imu_list': []}
    """
    acc_sensitivity = header_info.get("acc_sensitivity", 0.061)
    gyr_sensitivity = header_info.get("gyr_sensitivity", 4.375)
    imu_odr = header_info.get("imu_odr", 6667.0)
    
    # Calculate segment size: 1 minute of IMU samples
    n_imus_per_segment = round(imu_odr * 60)
    logger.info(f"Segment size: {n_imus_per_segment} IMU samples (≈1 minute at {imu_odr} Hz)")
    logger.info("Additional segmentation on IMU micros jumps: negative or > 1 second")
    
    pps_struct_size = PPS_STRUCT_SIZE
    gps_struct_size = GPS_STRUCT_SIZE
    imu_struct_size = IMU_STRUCT_SIZE
    
    # Initialize first segment
    segments = []
    current_segment = {'pps_list': [], 'gnss_list': [], 'imu_list': []}
    segments.append(current_segment)
    segment_imu_count = 0
    prev_imu_micros = None  # Track previous IMU micros for jump detection
    
    start_offset = 0
    idx = 0
    while idx < len(content):
        # Check if we need to start a new segment based on time
        if segment_imu_count >= n_imus_per_segment:
            logger.info(
                f"Starting segment {len(segments)} at byte {idx} "
                f"(segment {len(segments)-1} had {segment_imu_count} IMUs, "
                f"{len(current_segment['pps_list'])} PPS, {len(current_segment['gnss_list'])} GNSS) - TIME THRESHOLD"
            )
            current_segment = {'pps_list': [], 'gnss_list': [], 'imu_list': []}
            segments.append(current_segment)
            segment_imu_count = 0
            prev_imu_micros = None  # Reset for new segment
        
        if content[idx : idx + 4] == pps_marker:
            idx, should_break = process_pps_entry(
                content, idx, current_segment['pps_list'], current_segment['gnss_list'], current_segment['imu_list'],
                pps_marker, gps_marker, imu_marker, footer_marker,
                pps_struct_size
            )
            if should_break:
                break
                
        elif content[idx : idx + 4] == gps_marker:
            idx, should_break = process_gnss_entry(
                content, idx, current_segment['pps_list'], current_segment['gnss_list'], current_segment['imu_list'],
                pps_marker, gps_marker, imu_marker, footer_marker,
                gps_struct_size
            )
            if should_break:
                break
                
        elif content[idx : idx + 4] == imu_marker:
            idx, should_break, jump_detected = process_imu_entry(
                content, idx, current_segment['pps_list'], current_segment['gnss_list'], current_segment['imu_list'],
                pps_marker, gps_marker, imu_marker, footer_marker,
                imu_struct_size, acc_sensitivity, gyr_sensitivity,
                prev_imu_micros
            )
            if should_break:
                break
            
            # Check if jump was detected and we should start a new segment
            if jump_detected and len(current_segment['imu_list']) > 0:
                # Move the current IMU entry (which has the jump) to a new segment
                jumped_imu_entry = current_segment['imu_list'].pop()
                
                logger.info(
                    f"Starting segment {len(segments)} at byte {idx} "
                    f"(segment {len(segments)-1} had {segment_imu_count} IMUs, "
                    f"{len(current_segment['pps_list'])} PPS, {len(current_segment['gnss_list'])} GNSS) - MICROS JUMP"
                )
                
                # Start new segment with the jumped entry
                current_segment = {'pps_list': [], 'gnss_list': [], 'imu_list': [jumped_imu_entry]}
                segments.append(current_segment)
                segment_imu_count = 1
                prev_imu_micros = jumped_imu_entry.micros_reading
            else:
                # Normal processing
                segment_imu_count += 1
                if len(current_segment['imu_list']) > 0:
                    prev_imu_micros = current_segment['imu_list'][-1].micros_reading
                
        elif footer_marker in content[idx : idx + len(footer_marker) + 10]:
            logger.info("Found footer marker, stopping parsing")
            break
        else:
            idx += 1
    
    # Check if file ended properly
    footer_found = footer_marker in content[max(0, idx - 100) : idx + 100]
    
    # Calculate totals across all segments
    total_pps = sum(len(seg['pps_list']) for seg in segments)
    total_gnss = sum(len(seg['gnss_list']) for seg in segments)
    total_imu = sum(len(seg['imu_list']) for seg in segments)
    
    if not footer_found and idx >= len(content):
        logger.warning(f"Missing footer at end of file (byte {len(content)})")
        logger.error(
            f"File incomplete. Parsed {total_pps} PPS, "
            f"{total_gnss} GNSS, {total_imu} IMU entries before EOF"
        )
    elif not footer_found and idx < len(content):
        remaining_bytes = len(content) - idx
        logger.warning(
            f"Parsing stopped at byte {idx} ({remaining_bytes} bytes remaining, "
            f"{remaining_bytes / len(content) * 100:.1f}% of file unprocessed)"
        )
        logger.error(
            f"Unrecoverable corruption. Parsed {total_pps} PPS, "
            f"{total_gnss} GNSS, {total_imu} IMU before corruption"
        )
    
    # Log final parse statistics
    bytes_parsed = idx - start_offset
    total_entries = total_pps + total_gnss + total_imu
    
    logger.info(f"Created {len(segments)} segments")
    logger.info(f"Total: {total_pps} PPS, {total_gnss} GNSS, {total_imu} IMU entries")
    logger.info(f"Processed {bytes_parsed:,} bytes ({bytes_parsed / len(content) * 100:.1f}% of file)")
    if total_entries > 0:
        logger.info(f"Average {bytes_parsed / total_entries:.1f} bytes per entry")
    
    return segments



def save_decoded_data(
    segments: list[dict[str, list]],
    output_dir: Path,
    base_name: str,
    header_info: dict[str, Any],
    header_text: str,
    unwrap_stats: dict | None = None,
) -> dict[str, Path]:
    """Save decoded data to compressed numpy archive with segment naming.
    
    Args:
        segments: List of segment dicts, each with pps_list, gnss_list, imu_list
        output_dir: Directory to save files
        base_name: Base name for output file
        header_info: Dictionary of parsed header values
        header_text: Full header text string
        unwrap_stats: Optional unwrap statistics to include in return
        
    Returns:
        Dictionary with keys:
        - "file": Path to compressed .npz file
        - "unwrap_stats": Unwrap statistics (if provided)
    """
    output_files = {}
    
    # Prepare header data for storage
    header_string_array = np.array([header_text], dtype=object)
    
    save_dict = {
        "header_string": header_string_array,
        "number_of_segments": np.array([len(segments)]),
    }
    
    # Add individual header fields as separate arrays (without segment suffix)
    if "acc_sensitivity" in header_info:
        save_dict["acc_sensitivity"] = np.array([header_info["acc_sensitivity"]])
    if "gyr_sensitivity" in header_info:
        save_dict["gyr_sensitivity"] = np.array([header_info["gyr_sensitivity"]])
    if "imu_odr" in header_info:
        save_dict["imu_odr"] = np.array([header_info["imu_odr"]])
    if "gnss_rate" in header_info:
        save_dict["gnss_rate"] = np.array([header_info["gnss_rate"]])
    if "firmware_commit" in header_info:
        save_dict["firmware_commit"] = np.array([header_info["firmware_commit"]], dtype=object)
    
    # Save each segment with _segmentXXX naming
    for seg_idx, segment in enumerate(segments):
        seg_suffix = f"_segment{seg_idx:03d}"
        
        # Convert segment lists to arrays
        pps_array = np.array(segment['pps_list'], dtype=object)
        gnss_array = np.array(segment['gnss_list'], dtype=object)
        imu_array = np.array(segment['imu_list'], dtype=object)
        
        save_dict[f"pps{seg_suffix}"] = pps_array
        save_dict[f"gnss{seg_suffix}"] = gnss_array
        save_dict[f"imu{seg_suffix}"] = imu_array
    
    # Save as single compressed file
    npz_file = output_dir / f"{base_name}.npz"
    np.savez_compressed(npz_file, **save_dict)
    output_files["file"] = npz_file
    logger.info(f"Saved {len(segments)} segments to {npz_file} (compressed)")
    
    if unwrap_stats is not None:
        output_files["unwrap_stats"] = unwrap_stats
    
    return output_files


def load_and_combine_segments(npz_file: Path) -> dict[str, Any]:
    """Load segmented NPZ file and combine segments into single arrays.
    
    Args:
        npz_file: Path to the segmented .npz file
        
    Returns:
        Dictionary with combined arrays and header info:
        - 'pps': Combined PPS array
        - 'gnss': Combined GNSS array
        - 'imu': Combined IMU array
        - 'header_string': Header text
        - 'acc_sensitivity', 'gyr_sensitivity', etc.: Header fields
        - 'number_of_segments': Number of segments in original file
    """
    logger.info(f"Loading segmented data from {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    result = {}
    
    # Load number of segments
    number_of_segments = int(data['number_of_segments'][0])
    result['number_of_segments'] = number_of_segments
    logger.info(f"File contains {number_of_segments} segments")
    
    # Combine segments
    pps_segments = []
    gnss_segments = []
    imu_segments = []
    
    for seg_idx in range(number_of_segments):
        seg_suffix = f"_segment{seg_idx:03d}"
        pps_segments.append(data[f"pps{seg_suffix}"])
        gnss_segments.append(data[f"gnss{seg_suffix}"])
        imu_segments.append(data[f"imu{seg_suffix}"])
    
    # Concatenate all segments
    result['pps'] = np.concatenate(pps_segments) if pps_segments else np.array([])
    result['gnss'] = np.concatenate(gnss_segments) if gnss_segments else np.array([])
    result['imu'] = np.concatenate(imu_segments) if imu_segments else np.array([])
    
    logger.info(
        f"Combined: {len(result['pps'])} PPS, "
        f"{len(result['gnss'])} GNSS, {len(result['imu'])} IMU entries"
    )
    
    # Copy header info (non-segment data)
    for key in data.keys():
        if not key.startswith('pps') and not key.startswith('gnss') and not key.startswith('imu'):
            if key != 'number_of_segments':
                result[key] = data[key]
    
    return result


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
    """Decode a single data file and save to compressed numpy archive with segments.

    This function performs the complete decoding pipeline:
    1. Parses file header to extract sensor sensitivities
    2. Scans binary file for PPS, GNSS, and IMU entries in 1-minute segments
    3. For each segment independently:
       - Unwraps potentially wrapping counters and detects anomalies
       - Computes linear regression from PPS+GNSS to get UTC timestamps
       - Applies regression to all entries for synchronized timestamps
    4. Saves decoded data to compressed .npz file with segment naming

    Args:
        input_file: Path to input data file
        output_dir: Directory to save output files (defaults to same as input)
        show_plots: If True, display ASCII plots (e.g., PPS mismatch plot)
        pps_marker, gps_marker, imu_marker, footer_marker: Entry markers
        pps_struct_size: Size of PPS struct in bytes (default: 4)
        gps_struct_size: Size of GPS struct in bytes (default: 36)
        imu_struct_size: Size of IMU struct in bytes (default: 18)

    Returns:
        Dictionary with keys:
        - "file": Path to compressed .npz file
        - "unwrap_stats": Unwrap statistics with wrap/jump counts per segment

    Raises:
        AssertionError: If binary data structure doesn't match expected format
        struct.error: If binary unpacking fails
    """
    logger.info(f"Decoding file: {input_file}")

    if output_dir is None:
        output_dir = input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    header_info, header_text = parse_header(input_file, markers=(pps_marker, gps_marker, imu_marker))

    with open(input_file, "rb") as f:
        content = f.read()

    # Parse binary content into segments
    segments = parse_binary_content(
        content, header_info, pps_marker, gps_marker, imu_marker, footer_marker
    )

    # Process each segment independently with unwrap offset carryover
    all_unwrap_stats = {}
    
    # Initialize unwrap offsets and last raw values (carry over between segments)
    pps_micros_offset = 0
    pps_micros_prev_raw = None
    gnss_micros_offset = 0
    gnss_micros_prev_raw = None
    imu_micros_offset = 0
    imu_micros_prev_raw = None
    imu_counter_offset = 0
    imu_counter_prev_raw = None
    
    for seg_idx, segment in enumerate(segments):
        logger.info(f"Processing segment {seg_idx}...")
        
        pps_list = segment['pps_list']
        gnss_list = segment['gnss_list']
        imu_list = segment['imu_list']
        
        # Skip empty segments
        if not pps_list and not gnss_list and not imu_list:
            logger.warning(f"Segment {seg_idx} is empty, skipping")
            continue
        
        # Find minimum micros reading for this segment
        all_micros = (
            [p.micros_reading for p in pps_list] +
            [g.micros_reading for g in gnss_list] +
            [i.micros_reading for i in imu_list]
        )
        segment_min_micros = min(all_micros) if all_micros else 0
        
        # Unwrap potentially wrapping arrays and detect jumps (per segment with offset carryover)
        segment_unwrap_stats = {}

        # Process PPS micros_reading
        if pps_list:
            pps_micros = np.array([p.micros_reading for p in pps_list])
            pps_micros_unwrapped, pps_micros_wraps, pps_micros_jumps, pps_micros_offset, pps_micros_prev_raw = unwrap_array(
                pps_micros, max_value=2**32, initial_offset=pps_micros_offset, prev_raw_value=pps_micros_prev_raw
            )
            for i, pps in enumerate(pps_list):
                pps.micros_reading_unwrapped = int(pps_micros_unwrapped[i])

            segment_unwrap_stats["PPS"] = {
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
            gnss_micros_unwrapped, gnss_micros_wraps, gnss_micros_jumps, gnss_micros_offset, gnss_micros_prev_raw = unwrap_array(
                gnss_micros, max_value=2**32, initial_offset=gnss_micros_offset, prev_raw_value=gnss_micros_prev_raw
            )
            for i, gnss in enumerate(gnss_list):
                gnss.micros_reading_unwrapped = int(gnss_micros_unwrapped[i])

            segment_unwrap_stats["GNSS"] = {
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
            imu_micros_unwrapped, imu_micros_wraps, imu_micros_jumps, imu_micros_offset, imu_micros_prev_raw = unwrap_array(
                imu_micros, max_value=2**32, initial_offset=imu_micros_offset, prev_raw_value=imu_micros_prev_raw
            )

            imu_counter = np.array([i.counter for i in imu_list])
            imu_counter_unwrapped, imu_counter_wraps, imu_counter_jumps, imu_counter_offset, imu_counter_prev_raw = unwrap_array(
                imu_counter, max_value=2**16, jump_threshold=1, initial_offset=imu_counter_offset, prev_raw_value=imu_counter_prev_raw
            )

            for i, imu in enumerate(imu_list):
                imu.micros_reading_unwrapped = int(imu_micros_unwrapped[i])
                imu.counter_unwrapped = int(imu_counter_unwrapped[i])

            segment_unwrap_stats["IMU"] = {
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
        
        # Store unwrap stats for this segment
        all_unwrap_stats[f"segment_{seg_idx:03d}"] = segment_unwrap_stats

        # Compute PPS regression for this segment
        logger.info(f"Computing PPS to UTC timestamp regression for segment {seg_idx}...")
        regression = compute_pps_regression(pps_list, gnss_list, segment_min_micros)
        if regression is None:
            logger.warning(f"Skipping PPS regression for segment {seg_idx} due to insufficient data")
        else:
            slope, intercept = regression
            apply_pps_regression(pps_list, gnss_list, imu_list, slope, intercept)
            if pps_list:
                compute_pps_mismatch_statistics(pps_list)

        # Print summary statistics for this segment
        logger.info(
            f"Segment {seg_idx}: {len(pps_list)} PPS, "
            f"{len(gnss_list)} GNSS, {len(imu_list)} IMU entries"
        )
        
        # Apply outlier detection to physical variables
        # IMU acceleration and gyroscope
        if imu_list and len(imu_list) > 4:
            acc_x_values = np.array([i.acc_x_mg for i in imu_list])
            acc_y_values = np.array([i.acc_y_mg for i in imu_list])
            acc_z_values = np.array([i.acc_z_mg for i in imu_list])
            gyr_x_values = np.array([i.gyr_x_mdps for i in imu_list])
            gyr_y_values = np.array([i.gyr_y_mdps for i in imu_list])
            gyr_z_values = np.array([i.gyr_z_mdps for i in imu_list])
            
            acc_x_outliers = detect_outliers_stdcheck(acc_x_values)
            acc_y_outliers = detect_outliers_stdcheck(acc_y_values)
            acc_z_outliers = detect_outliers_stdcheck(acc_z_values)
            gyr_x_outliers = detect_outliers_stdcheck(gyr_x_values)
            gyr_y_outliers = detect_outliers_stdcheck(gyr_y_values)
            gyr_z_outliers = detect_outliers_stdcheck(gyr_z_values)
            
            # Flag outliers in the data structures
            apply_outlier_flags(imu_list, 'acc_x_mg_stdchecked', acc_x_outliers)
            apply_outlier_flags(imu_list, 'acc_y_mg_stdchecked', acc_y_outliers)
            apply_outlier_flags(imu_list, 'acc_z_mg_stdchecked', acc_z_outliers)
            apply_outlier_flags(imu_list, 'gyr_x_mdps_stdchecked', gyr_x_outliers)
            apply_outlier_flags(imu_list, 'gyr_y_mdps_stdchecked', gyr_y_outliers)
            apply_outlier_flags(imu_list, 'gyr_z_mdps_stdchecked', gyr_z_outliers)
            
            n_acc_outliers = len(acc_x_outliers) + len(acc_y_outliers) + len(acc_z_outliers)
            n_gyr_outliers = len(gyr_x_outliers) + len(gyr_y_outliers) + len(gyr_z_outliers)
            if n_acc_outliers > 0 or n_gyr_outliers > 0:
                logger.info(
                    f"Segment {seg_idx} IMU outliers detected: "
                    f"{n_acc_outliers} acceleration, {n_gyr_outliers} gyroscope"
                )
        
        # GNSS position and velocity
        if gnss_list and len(gnss_list) > 4:
            lat_values = np.array([g.latitude_dd for g in gnss_list])
            lon_values = np.array([g.longitude_dd for g in gnss_list])
            vel_n_values = np.array([g.ned_vel_north_mmps for g in gnss_list])
            vel_e_values = np.array([g.ned_vel_east_mmps for g in gnss_list])
            vel_d_values = np.array([g.ned_vel_down_mmps for g in gnss_list])
            
            lat_outliers = detect_outliers_stdcheck(lat_values)
            lon_outliers = detect_outliers_stdcheck(lon_values)
            vel_n_outliers = detect_outliers_stdcheck(vel_n_values)
            vel_e_outliers = detect_outliers_stdcheck(vel_e_values)
            vel_d_outliers = detect_outliers_stdcheck(vel_d_values)
            
            # Flag outliers in the data structures
            apply_outlier_flags(gnss_list, 'latitude_dd_stdchecked', lat_outliers)
            apply_outlier_flags(gnss_list, 'longitude_dd_stdchecked', lon_outliers)
            apply_outlier_flags(gnss_list, 'ned_vel_north_mmps_stdchecked', vel_n_outliers)
            apply_outlier_flags(gnss_list, 'ned_vel_east_mmps_stdchecked', vel_e_outliers)
            apply_outlier_flags(gnss_list, 'ned_vel_down_mmps_stdchecked', vel_d_outliers)
            
            n_pos_outliers = len(lat_outliers) + len(lon_outliers)
            n_vel_outliers = len(vel_n_outliers) + len(vel_e_outliers) + len(vel_d_outliers)
            if n_pos_outliers > 0 or n_vel_outliers > 0:
                logger.info(
                    f"Segment {seg_idx} GNSS outliers detected: "
                    f"{n_pos_outliers} position, {n_vel_outliers} velocity"
                )

    # Print overall summary statistics
    total_pps = sum(len(seg['pps_list']) for seg in segments)
    total_gnss = sum(len(seg['gnss_list']) for seg in segments)
    total_imu = sum(len(seg['imu_list']) for seg in segments)
    logger.info(
        f"Total across all segments: {total_pps} PPS, "
        f"{total_gnss} GNSS, {total_imu} IMU entries"
    )

    # Save results and return file paths with unwrap stats
    return save_decoded_data(
        segments, output_dir, input_file.stem,
        header_info, header_text, all_unwrap_stats
    )
