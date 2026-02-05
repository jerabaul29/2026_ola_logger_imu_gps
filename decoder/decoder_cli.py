#!/usr/bin/env python3
"""CLI tool for decoding and visualizing OLA logger data files."""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from decoder import decode_file


@click.command()
@click.option(
    "-p",
    "--path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the data file to decode",
)
def main(path: Path) -> None:
    """Decode OLA logger data file and create matplotlib visualizations.

    This CLI tool decodes the specified data file and generates several plots:
    - 3-axis acceleration (mg)
    - 3-axis gyroscope (mdps)
    - GNSS coordinates (latitude/longitude)
    - Time differences between consecutive IMU and GNSS entries
    - PPS mismatch (UTC regression vs closest second)
    - IMU counter vs entry number
    """
    logger.info(f"Decoding file: {path}")

    # Decode the file (no ASCII plots)
    output_files = decode_file(path, show_plots=False)

    # Extract unwrap stats
    unwrap_stats = output_files.get("unwrap_stats", None)

    # Load the decoded data
    logger.info("Loading decoded data for visualization...")
    pps_data = np.load(output_files["pps"], allow_pickle=True)
    gnss_data = np.load(output_files["gnss"], allow_pickle=True)
    imu_data = np.load(output_files["imu"], allow_pickle=True)

    logger.info(f"Loaded {len(pps_data)} PPS, {len(gnss_data)} GNSS, {len(imu_data)} IMU entries")

    # Create visualizations
    logger.info("Creating matplotlib visualizations...")

    # Plot 1: 3-axis acceleration
    plot_imu_acceleration(imu_data)

    # Plot 2: 3-axis gyroscope
    plot_imu_gyroscope(imu_data)

    # Plot 3: GNSS coordinates
    plot_gnss_coordinates(gnss_data)

    # Plot 4: Time differences between consecutive entries
    plot_time_differences(imu_data, gnss_data)

    # Plot 5: PPS mismatch
    plot_pps_mismatch(pps_data)

    # Plot 6: IMU counter vs entry number
    plot_imu_counter(imu_data, unwrap_stats)

    logger.success("All plots created! Close plot windows to exit.")
    plt.show()


def plot_imu_acceleration(imu_data: np.ndarray) -> None:
    """Plot 3-axis acceleration data.

    Args:
        imu_data: Array of IMUReading objects
    """
    if len(imu_data) == 0:
        logger.warning("No IMU data to plot")
        return

    # Extract time and acceleration data
    time_utc = np.array([imu.utc_timestamp_from_pps_regression for imu in imu_data])
    time_relative = time_utc - time_utc[0]  # Relative time in seconds

    acc_x = np.array([imu.acc_x_mg for imu in imu_data])
    acc_y = np.array([imu.acc_y_mg for imu in imu_data])
    acc_z = np.array([imu.acc_z_mg for imu in imu_data])

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("IMU Acceleration (3-axis)", fontsize=14, fontweight="bold")

    # Plot each axis
    axes[0].plot(time_relative, acc_x, "r-", linewidth=0.5, label="Acc X")
    axes[0].set_ylabel("Acc X (mg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_relative, acc_y, "g-", linewidth=0.5, label="Acc Y")
    axes[1].set_ylabel("Acc Y (mg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time_relative, acc_z, "b-", linewidth=0.5, label="Acc Z")
    axes[2].set_ylabel("Acc Z (mg)")
    axes[2].set_xlabel("Time since start (seconds)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    logger.info("Created acceleration plot")


def plot_imu_gyroscope(imu_data: np.ndarray) -> None:
    """Plot 3-axis gyroscope data.

    Args:
        imu_data: Array of IMUReading objects
    """
    if len(imu_data) == 0:
        logger.warning("No IMU data to plot")
        return

    # Extract time and gyroscope data
    time_utc = np.array([imu.utc_timestamp_from_pps_regression for imu in imu_data])
    time_relative = time_utc - time_utc[0]  # Relative time in seconds

    gyr_x = np.array([imu.gyr_x_mdps for imu in imu_data])
    gyr_y = np.array([imu.gyr_y_mdps for imu in imu_data])
    gyr_z = np.array([imu.gyr_z_mdps for imu in imu_data])

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("IMU Gyroscope (3-axis)", fontsize=14, fontweight="bold")

    # Plot each axis
    axes[0].plot(time_relative, gyr_x, "r-", linewidth=0.5, label="Gyr X")
    axes[0].set_ylabel("Gyr X (mdps)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_relative, gyr_y, "g-", linewidth=0.5, label="Gyr Y")
    axes[1].set_ylabel("Gyr Y (mdps)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time_relative, gyr_z, "b-", linewidth=0.5, label="Gyr Z")
    axes[2].set_ylabel("Gyr Z (mdps)")
    axes[2].set_xlabel("Time since start (seconds)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    logger.info("Created gyroscope plot")


def plot_gnss_coordinates(gnss_data: np.ndarray) -> None:
    """Plot GNSS coordinates (latitude and longitude over time).

    Args:
        gnss_data: Array of GNSSReading objects
    """
    if len(gnss_data) == 0:
        logger.warning("No GNSS data to plot")
        return

    # Extract time and position data
    time_utc = np.array([gnss.utc_timestamp_from_pps_regression for gnss in gnss_data])
    time_relative = time_utc - time_utc[0]  # Relative time in seconds

    latitude = np.array([gnss.latitude_dd for gnss in gnss_data])
    longitude = np.array([gnss.longitude_dd for gnss in gnss_data])

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("GNSS Coordinates", fontsize=14, fontweight="bold")

    # Plot latitude
    axes[0].plot(time_relative, latitude, "b-", linewidth=1, label="Latitude")
    axes[0].set_ylabel("Latitude (degrees)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot longitude
    axes[1].plot(time_relative, longitude, "r-", linewidth=1, label="Longitude")
    axes[1].set_ylabel("Longitude (degrees)")
    axes[1].set_xlabel("Time since start (seconds)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    logger.info("Created GNSS coordinates plot")


def plot_time_differences(imu_data: np.ndarray, gnss_data: np.ndarray) -> None:
    """Plot time differences between consecutive entries.

    Computes and plots the time differences (dt) between consecutive
    measurements for both IMU and GNSS, using both micros timestamps
    and UTC timestamps from the PPS regression.

    Args:
        imu_data: Array of IMUReading objects
        gnss_data: Array of GNSSReading objects
    """
    if len(imu_data) < 2 and len(gnss_data) < 2:
        logger.warning("Not enough data to compute time differences")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Time Differences Between Consecutive Entries", fontsize=14, fontweight="bold")

    # IMU time differences from micros
    if len(imu_data) >= 2:
        micros_imu = np.array([imu.micros_reading for imu in imu_data])
        dt_micros_imu = np.diff(micros_imu) / 1000.0  # Convert to milliseconds

        axes[0, 0].plot(dt_micros_imu, "b.", markersize=1, alpha=0.5)
        axes[0, 0].set_ylabel("Δt (ms)")
        axes[0, 0].set_xlabel("Sample index")
        axes[0, 0].set_title("IMU: Time diff from micros()")
        axes[0, 0].grid(True, alpha=0.3)

        mean_dt = np.mean(dt_micros_imu)
        std_dt = np.std(dt_micros_imu)
        axes[0, 0].axhline(mean_dt, color="r", linestyle="--", linewidth=1,
                           label=f"Mean: {mean_dt:.3f} ms")
        axes[0, 0].legend()

        logger.info(f"IMU dt from micros: mean={mean_dt:.3f} ms, std={std_dt:.3f} ms")

    # IMU time differences from UTC regression
    if len(imu_data) >= 2:
        utc_imu = np.array([imu.utc_timestamp_from_pps_regression for imu in imu_data])
        dt_utc_imu = np.diff(utc_imu) * 1000  # Convert to milliseconds

        axes[0, 1].plot(dt_utc_imu, "b.", markersize=1, alpha=0.5)
        axes[0, 1].set_ylabel("Δt (ms)")
        axes[0, 1].set_xlabel("Sample index")
        axes[0, 1].set_title("IMU: Time diff from UTC regression")
        axes[0, 1].grid(True, alpha=0.3)

        mean_dt = np.mean(dt_utc_imu)
        std_dt = np.std(dt_utc_imu)
        axes[0, 1].axhline(mean_dt, color="r", linestyle="--", linewidth=1,
                           label=f"Mean: {mean_dt:.3f} ms")
        axes[0, 1].legend()

        logger.info(f"IMU dt from UTC: mean={mean_dt:.3f} ms, std={std_dt:.3f} ms")

    # GNSS time differences from micros
    if len(gnss_data) >= 2:
        micros_gnss = np.array([gnss.micros_reading for gnss in gnss_data])
        dt_micros_gnss = np.diff(micros_gnss) / 1000.0  # Convert to milliseconds

        axes[1, 0].plot(dt_micros_gnss, "g.", markersize=2, alpha=0.5)
        axes[1, 0].set_ylabel("Δt (ms)")
        axes[1, 0].set_xlabel("Sample index")
        axes[1, 0].set_title("GNSS: Time diff from micros()")
        axes[1, 0].grid(True, alpha=0.3)

        mean_dt = np.mean(dt_micros_gnss)
        std_dt = np.std(dt_micros_gnss)
        axes[1, 0].axhline(mean_dt, color="r", linestyle="--", linewidth=1,
                           label=f"Mean: {mean_dt:.1f} ms")
        axes[1, 0].legend()

        logger.info(f"GNSS dt from micros: mean={mean_dt:.1f} ms, std={std_dt:.1f} ms")

    # GNSS time differences from UTC regression
    if len(gnss_data) >= 2:
        utc_gnss = np.array([gnss.utc_timestamp_from_pps_regression for gnss in gnss_data])
        dt_utc_gnss = np.diff(utc_gnss) * 1000  # Convert to milliseconds

        axes[1, 1].plot(dt_utc_gnss, "g.", markersize=2, alpha=0.5)
        axes[1, 1].set_ylabel("Δt (ms)")
        axes[1, 1].set_xlabel("Sample index")
        axes[1, 1].set_title("GNSS: Time diff from UTC regression")
        axes[1, 1].grid(True, alpha=0.3)

        mean_dt = np.mean(dt_utc_gnss)
        std_dt = np.std(dt_utc_gnss)
        axes[1, 1].axhline(mean_dt, color="r", linestyle="--", linewidth=1,
                           label=f"Mean: {mean_dt:.1f} ms")
        axes[1, 1].legend()

        logger.info(f"GNSS dt from UTC: mean={mean_dt:.1f} ms, std={std_dt:.1f} ms")

    plt.tight_layout()
    logger.info("Created time differences plot")


def plot_imu_counter(imu_data: np.ndarray, unwrap_stats: dict | None = None) -> None:
    """Plot IMU counter as a function of entry number.

    Creates two subplots:
    - Left: Raw counter values
    - Right: Unwrapped counter with anomalies highlighted

    Args:
        imu_data: Array of IMUReading objects
        unwrap_stats: Optional unwrap statistics from decoder
    """
    if len(imu_data) == 0:
        logger.warning("No IMU data to plot")
        return

    # Extract counter values (raw and unwrapped)
    counters = np.array([imu.counter for imu in imu_data], dtype=np.int64)
    unwrapped_counters = np.array([imu.counter_unwrapped for imu in imu_data], dtype=np.int64)
    entry_numbers = np.arange(len(imu_data))

    # Get wrap and jump counts from unwrap_stats if available
    if unwrap_stats and "IMU" in unwrap_stats and "counter" in unwrap_stats["IMU"]:
        num_wraps = unwrap_stats["IMU"]["counter"]["wraps"]
        wrap_indices_array = unwrap_stats["IMU"]["counter"].get("wrap_indices")
        jump_indices_array = unwrap_stats["IMU"]["counter"].get("jump_indices")
        wrap_indices = list(wrap_indices_array) if wrap_indices_array is not None else []
        anomaly_indices = list(jump_indices_array) if jump_indices_array is not None else []
        num_anomalies = len(anomaly_indices)
    else:
        # Fallback: detect anomalies from unwrapped data
        num_wraps = 0
        wrap_indices = []
        anomaly_indices = []
        for i in range(1, len(unwrapped_counters)):
            increment = unwrapped_counters[i] - unwrapped_counters[i-1]
            if increment != 1:
                anomaly_indices.append(i)
                # Detect wraps: large jumps in unwrapped (but this is approximate)
                if increment > 60000:
                    num_wraps += 1
                    wrap_indices.append(i)
        num_anomalies = len(anomaly_indices)

    logger.info(f"IMU counter: {num_wraps} wraps detected, "
                f"{num_anomalies} anomalous samples (missed data)")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("IMU Counter vs Entry Number", fontsize=14, fontweight="bold")

    # Left plot: raw counter
    axes[0].plot(entry_numbers, counters, "b-", linewidth=0.8, label="Raw Counter")
    axes[0].set_xlabel("Entry Number")
    axes[0].set_ylabel("Counter Value")
    axes[0].set_title(f"Raw Counter (uint16): {num_wraps} wraps, {num_anomalies} jumps (anomalous)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right plot: unwrapped counter with wraps and anomalies
    axes[1].plot(entry_numbers, unwrapped_counters, "b-", linewidth=0.8,
                 label="Unwrapped Counter")

    # Highlight wraps with green crosses
    if wrap_indices:
        axes[1].plot(entry_numbers[wrap_indices],
                     unwrapped_counters[wrap_indices],
                     "gx", markersize=10, markeredgewidth=2,
                     label=f"Wraps ({num_wraps})")

    # Highlight anomalies with red crosses
    if anomaly_indices:
        axes[1].plot(entry_numbers[anomaly_indices],
                     unwrapped_counters[anomaly_indices],
                     "rx", markersize=8, markeredgewidth=2,
                     label=f"Anomalies ({num_anomalies})")

    axes[1].set_xlabel("Entry Number")
    axes[1].set_ylabel("Unwrapped Counter Value")
    axes[1].set_title(f"Unwrapped Counter: {num_wraps} wraps, {num_anomalies} jumps (anomalous)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    logger.info("Created IMU counter plot")


def plot_pps_mismatch(pps_data: np.ndarray) -> None:
    """Plot PPS mismatch between UTC regression and closest second.

    Computes and plots the mismatch between the UTC datetime from linear
    regression and the closest UTC second for each PPS entry.

    Args:
        pps_data: Array of PPSFix objects
    """
    if len(pps_data) == 0:
        logger.warning("No PPS data to plot")
        return

    # Check if regression was computed
    if pps_data[0].utc_timestamp_from_pps_regression is None:
        logger.warning("PPS regression not computed, skipping mismatch plot")
        return

    # Compute mismatch for each PPS entry
    mismatches = []
    time_points = []

    first_pps_utc = pps_data[0].utc_timestamp_from_pps_regression

    for pps in pps_data:
        utc_timestamp = pps.utc_timestamp_from_pps_regression
        closest_second = round(utc_timestamp)
        mismatch = utc_timestamp - closest_second
        mismatches.append(mismatch * 1000)  # Convert to milliseconds
        time_points.append(utc_timestamp - first_pps_utc)  # Time since start

    mismatches = np.array(mismatches)
    time_points = np.array(time_points)

    # Compute statistics
    max_mismatch = np.max(np.abs(mismatches))
    mean_mismatch = np.mean(mismatches)
    rms_mismatch = np.sqrt(np.mean(mismatches ** 2))

    logger.info(f"PPS Mismatch: max={max_mismatch:.3f} ms, "
                f"mean={mean_mismatch:.3f} ms, rms={rms_mismatch:.3f} ms")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("PPS UTC Mismatch: Regression vs Closest Second",
                 fontsize=14, fontweight="bold")

    ax.plot(time_points, mismatches, "b-", linewidth=0.8, label="UTC mismatch")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time since first PPS (seconds)")
    ax.set_ylabel("UTC mismatch (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add statistics text
    stats_text = (f"Max: {max_mismatch:.3f} ms\n"
                  f"Mean: {mean_mismatch:.3f} ms\n"
                  f"RMS: {rms_mismatch:.3f} ms")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    logger.info("Created PPS mismatch plot")


if __name__ == "__main__":
    main()
