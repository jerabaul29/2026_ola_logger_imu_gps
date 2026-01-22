"""Tests for the decoder module."""

import struct
from pathlib import Path

import numpy as np
import pytest

from decoder import (
    GNSSReading,
    IMUReading,
    PPSFix,
    decode_file,
    parse_gnss_entry,
    parse_header,
    parse_imu_entry,
    parse_pps_entry,
)


def test_parse_pps_entry():
    """Test parsing a PPS entry."""
    data = struct.pack("<I", 12345)
    pps = parse_pps_entry(data)
    assert isinstance(pps, PPSFix)
    assert pps.millis_reading == 12345


def test_parse_gnss_entry():
    """Test parsing a GNSS entry."""
    data = struct.pack(
        "<IiiiIiiiB",
        10000,
        123456789,
        -987654321,
        1705000000,
        500000,
        100,
        -50,
        25,
        3,
    )
    gnss = parse_gnss_entry(data)
    assert isinstance(gnss, GNSSReading)
    assert gnss.millis_reading == 10000
    assert gnss.latitude == 123456789
    assert gnss.longitude == -987654321
    assert gnss.posix_timestamp == 1705000000
    assert gnss.microseconds == 500000
    assert gnss.ned_vel_north == 100
    assert gnss.ned_vel_east == -50
    assert gnss.ned_vel_down == 25
    assert gnss.fix_type == 3


def test_parse_imu_entry():
    """Test parsing an IMU entry."""
    data = struct.pack("<Ihhhhhh", 5000, 100, -200, 300, 1000, -2000, 3000)
    imu = parse_imu_entry(data, acc_sensitivity=0.061, gyr_sensitivity=4.375)
    assert isinstance(imu, IMUReading)
    assert imu.millis_reading == 5000
    assert imu.acc_x == 100
    assert imu.acc_y == -200
    assert imu.acc_z == 300
    assert imu.gyr_x == 1000
    assert imu.gyr_y == -2000
    assert imu.gyr_z == 3000
    assert abs(imu.acc_x_mg - 100 * 0.061) < 1e-6
    assert abs(imu.acc_y_mg - (-200) * 0.061) < 1e-6
    assert abs(imu.acc_z_mg - 300 * 0.061) < 1e-6
    assert abs(imu.gyr_x_mdps - 1000 * 4.375) < 1e-6
    assert abs(imu.gyr_y_mdps - (-2000) * 4.375) < 1e-6
    assert abs(imu.gyr_z_mdps - 3000 * 4.375) < 1e-6


def test_parse_header(tmp_path):
    """Test parsing the file header."""
    test_file = tmp_path / "test_header.dat"
    header_content = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: 391b428a3e869543ebd2caf1626f845730858f8b
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""
    test_file.write_text(header_content)

    header_info = parse_header(test_file)
    assert header_info["acc_sensitivity"] == 0.061
    assert header_info["gyr_sensitivity"] == 4.375
    assert header_info["imu_odr"] == 417.0
    assert header_info["gnss_rate"] == 10.0
    assert header_info["firmware_commit"] == "391b428a3e869543ebd2caf1626f845730858f8b"


def test_decode_file_with_real_data():
    """Test decoding with the real data file if it exists."""
    test_file = Path("DATA_BOOT_0055_TIME_20260120T211500.dat")
    if not test_file.exists():
        pytest.skip("Real data file not found")

    output_files = decode_file(test_file)

    assert "pps" in output_files
    assert "gnss" in output_files
    assert "imu" in output_files

    pps_data = np.load(output_files["pps"], allow_pickle=True)
    gnss_data = np.load(output_files["gnss"], allow_pickle=True)
    imu_data = np.load(output_files["imu"], allow_pickle=True)

    assert len(pps_data) > 0
    assert len(gnss_data) > 0
    assert len(imu_data) > 0

    assert isinstance(pps_data[0], PPSFix)
    assert isinstance(gnss_data[0], GNSSReading)
    assert isinstance(imu_data[0], IMUReading)

    for output_file in output_files.values():
        if output_file.exists():
            output_file.unlink()


def test_decode_file_synthetic(tmp_path):
    """Test decoding with synthetic data."""
    test_file = tmp_path / "test_data.dat"

    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: test123
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""

    pps_entry = b"\nPPS" + struct.pack("<I", 1000)
    gnss_entry = b"\nGPS" + struct.pack(
        "<IiiiIiiiB", 2000, 12345678, -87654321, 1705000000, 0, 10, 20, 5, 3
    )
    imu_entry = b"\nIMU" + struct.pack("<Ihhhhhh", 3000, 100, 200, 300, 50, 100, 150)
    footer = b"\n\nLog stop OLA ISM330DHCX SAM-M10Q logger\n"

    with open(test_file, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(pps_entry)
        f.write(gnss_entry)
        f.write(imu_entry)
        f.write(footer)

    output_files = decode_file(test_file, output_dir=tmp_path)

    assert "pps" in output_files
    assert "gnss" in output_files
    assert "imu" in output_files

    pps_data = np.load(output_files["pps"], allow_pickle=True)
    gnss_data = np.load(output_files["gnss"], allow_pickle=True)
    imu_data = np.load(output_files["imu"], allow_pickle=True)

    assert len(pps_data) == 1
    assert len(gnss_data) == 1
    assert len(imu_data) == 1

    assert pps_data[0].millis_reading == 1000
    assert gnss_data[0].millis_reading == 2000
    assert gnss_data[0].latitude == 12345678
    assert imu_data[0].millis_reading == 3000
    assert imu_data[0].acc_x == 100


def test_parse_entry_assertions():
    """Test that assertions catch incorrect data sizes."""
    # Test PPS assertion
    with pytest.raises(AssertionError, match="PPS data size mismatch"):
        parse_pps_entry(b"123")  # Too short

    with pytest.raises(AssertionError, match="PPS data size mismatch"):
        parse_pps_entry(b"12345")  # Too long

    # Test GNSS assertion
    with pytest.raises(AssertionError, match="GNSS data size mismatch"):
        parse_gnss_entry(b"1" * 30)  # Too short

    with pytest.raises(AssertionError, match="GNSS data size mismatch"):
        parse_gnss_entry(b"1" * 40)  # Too long

    # Test IMU assertion
    with pytest.raises(AssertionError, match="IMU data size mismatch"):
        parse_imu_entry(b"1" * 10)  # Too short

    with pytest.raises(AssertionError, match="IMU data size mismatch"):
        parse_imu_entry(b"1" * 20)  # Too long
