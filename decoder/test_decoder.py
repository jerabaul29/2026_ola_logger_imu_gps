"""Tests for the decoder module."""

import struct
from pathlib import Path

import numpy as np
import pytest

from decoder import (
    GNSSReading,
    IMUReading,
    PPSFix,
    compute_pps_regression,
    decode_file,
    parse_gnss_entry,
    parse_header,
    parse_imu_entry,
    parse_pps_entry,
    unwrap_micros,
)


def test_parse_pps_entry():
    """Test parsing a PPS entry."""
    data = struct.pack("<I", 12345)
    pps = parse_pps_entry(data)
    assert isinstance(pps, PPSFix)
    assert pps.micros_reading == 12345


def test_parse_gnss_entry():
    """Test parsing a GNSS entry."""
    packed = struct.pack(
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
    ) + b"\x00\x00\x00"  # Add 3 bytes padding to reach 36 bytes
    gnss = parse_gnss_entry(packed)
    assert isinstance(gnss, GNSSReading)
    assert gnss.micros_reading == 10000
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
    data = struct.pack("<IHhhhhhh", 5000, 42, 100, -200, 300, 1000, -2000, 3000)
    imu = parse_imu_entry(data, acc_sensitivity=0.061, gyr_sensitivity=4.375)
    assert isinstance(imu, IMUReading)
    assert imu.micros_reading == 5000
    assert imu.counter == 42
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
    test_file = Path("DATA_BOOT_0000_TIME_20260204T193000.dat")
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
    
    # Verify new format fields exist
    assert hasattr(pps_data[0], 'micros_reading')
    assert hasattr(gnss_data[0], 'micros_reading')
    assert hasattr(imu_data[0], 'micros_reading')
    assert hasattr(imu_data[0], 'counter')
    
    # Verify counter increments (check first 10 entries)
    for i in range(1, min(10, len(imu_data))):
        # Allow for wrapping at uint16 max
        expected = (imu_data[i-1].counter + 1) % (2**16)
        assert imu_data[i].counter == expected or imu_data[i].counter == 0, \
            f"Counter not incrementing correctly at index {i}"
    
    # Verify micros timestamps are increasing
    for i in range(1, min(100, len(imu_data))):
        assert imu_data[i].micros_reading >= imu_data[i-1].micros_reading, \
            f"Micros not increasing at index {i}"

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
    gnss_entry = (
        b"\nGPS"
        + struct.pack(
            "<IiiiIiiiB", 2000, 12345678, -87654321, 1705000000, 0, 10, 20, 5, 3
        )
        + b"\x00\x00\x00"
    )  # pad to 36 bytes (33 struct + 3 padding)
    # IMU with padding: 18-byte struct + 2-byte padding
    imu_entry = b"\nIMU" + struct.pack("<IHhhhhhh", 3000, 0, 100, 200, 300, 50, 100, 150) + b"\x00\x00"
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

    assert pps_data[0].micros_reading == 1000
    assert gnss_data[0].micros_reading == 2000
    assert gnss_data[0].latitude == 12345678
    assert imu_data[0].micros_reading == 3000
    assert imu_data[0].counter == 0
    assert imu_data[0].acc_x == 100


def test_parse_entry_assertions():
    """Test that assertions catch incorrect data sizes."""
    # Test PPS assertion
    with pytest.raises(AssertionError, match="PPS data size mismatch"):
        parse_pps_entry(b"123")  # Too short

    with pytest.raises(AssertionError, match="PPS data size mismatch"):
        parse_pps_entry(b"12345")  # Too long

    # Test GNSS assertion - should expect exactly 36 bytes (33 struct + 3 padding)
    with pytest.raises(AssertionError, match="GNSS data size mismatch"):
        parse_gnss_entry(b"1" * 33)  # Too short (missing padding)

    with pytest.raises(AssertionError, match="GNSS data size mismatch"):
        parse_gnss_entry(b"1" * 40)  # Too long

    # Test IMU assertion - now expects 18 bytes (4+2+6*2)
    with pytest.raises(AssertionError, match="IMU data size mismatch"):
        parse_imu_entry(b"1" * 10)  # Too short

    with pytest.raises(AssertionError, match="IMU data size mismatch"):
        parse_imu_entry(b"1" * 20)  # Too long


def test_unwrap_micros_no_wrapping():
    """Test unwrap_micros with no wrapping."""
    micros = [1000, 2000, 3000, 4000]
    unwrapped = unwrap_micros(micros)
    assert unwrapped == [1000, 2000, 3000, 4000]


def test_unwrap_micros_with_wrapping():
    """Test unwrap_micros with wrapping at uint32_t boundary."""
    UINT32_MAX = 2**32
    # Simulate wrap: values go from near max to near zero
    micros = [UINT32_MAX - 1000, UINT32_MAX - 500, 100, 500]
    unwrapped = unwrap_micros(micros)

    # After unwrapping, values should be monotonic
    assert unwrapped[0] == UINT32_MAX - 1000
    assert unwrapped[1] == UINT32_MAX - 500
    assert unwrapped[2] == UINT32_MAX + 100
    assert unwrapped[3] == UINT32_MAX + 500


def test_unwrap_micros_empty():
    """Test unwrap_micros with empty list."""
    assert unwrap_micros([]) == []


def test_compute_pps_regression():
    """Test PPS to UTC regression computation."""
    # Create synthetic PPS and GNSS data with known relationship
    # micros = slope * utc + intercept
    # For simplicity: 1000000 us = 1 second UTC starting at t=1000000
    pps_list = [
        PPSFix(micros_reading=1000000),
        PPSFix(micros_reading=2000000),
        PPSFix(micros_reading=3000000),
    ]

    gnss_list = [
        GNSSReading(
            micros_reading=1000000, latitude=0, longitude=0,
            posix_timestamp=1, microseconds=0,
            ned_vel_north=0, ned_vel_east=0, ned_vel_down=0, fix_type=3,
            latitude_dd=0.0, longitude_dd=0.0,
            ned_vel_north_mmps=0, ned_vel_east_mmps=0, ned_vel_down_mmps=0,
            datetime_utc=None
        ),
        GNSSReading(
            micros_reading=2000000, latitude=0, longitude=0,
            posix_timestamp=2, microseconds=0,
            ned_vel_north=0, ned_vel_east=0, ned_vel_down=0, fix_type=3,
            latitude_dd=0.0, longitude_dd=0.0,
            ned_vel_north_mmps=0, ned_vel_east_mmps=0, ned_vel_down_mmps=0,
            datetime_utc=None
        ),
        GNSSReading(
            micros_reading=3000000, latitude=0, longitude=0,
            posix_timestamp=3, microseconds=0,
            ned_vel_north=0, ned_vel_east=0, ned_vel_down=0, fix_type=3,
            latitude_dd=0.0, longitude_dd=0.0,
            ned_vel_north_mmps=0, ned_vel_east_mmps=0, ned_vel_down_mmps=0,
            datetime_utc=None
        ),
    ]

    regression = compute_pps_regression(pps_list, gnss_list)
    assert regression is not None
    slope, intercept = regression

    # Expected: utc = slope * micros + intercept
    # With our data: utc=1 at micros=1000000, utc=2 at micros=2000000, etc.
    # slope should be 1e-6 (1 second per 1000000 micros)
    assert abs(slope - 1e-6) < 1e-9
    # intercept should be 0 (utc = 1e-6 * micros + 0)
    assert abs(intercept - 0.0) < 1e-6


def test_imu_padding_handling(tmp_path):
    """Test that IMU entries with 2-byte padding are parsed correctly."""
    test_file = tmp_path / "test_padding.dat"
    
    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: test_padding
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 208.00
GNSS update rate (Hz): 10
"""
    
    # Create multiple IMU entries with padding to test sequential parsing
    imu_entries = []
    for i in range(5):
        micros = 1000000 + i * 5000  # ~5ms between samples
        counter = 100 + i
        # 18-byte struct + 2-byte padding
        imu_entry = b"\nIMU" + struct.pack("<IHhhhhhh", 
                                           micros, counter,
                                           100+i, 200+i, 300+i,
                                           50+i, 100+i, 150+i) + b"\x00\x00"
        imu_entries.append(imu_entry)
    
    footer = b"\n\nLog stop OLA ISM330DHCX SAM-M10Q logger\n"
    
    with open(test_file, "wb") as f:
        f.write(header.encode("utf-8"))
        for entry in imu_entries:
            f.write(entry)
        f.write(footer)
    
    # Decode the file
    output_files = decode_file(test_file, output_dir=tmp_path)
    imu_data = np.load(output_files["imu"], allow_pickle=True)
    
    # Should have parsed all 5 entries
    assert len(imu_data) == 5
    
    # Verify each entry
    for i in range(5):
        assert imu_data[i].micros_reading == 1000000 + i * 5000
        assert imu_data[i].counter == 100 + i
        assert imu_data[i].acc_x == 100 + i
        assert imu_data[i].acc_y == 200 + i
        assert imu_data[i].acc_z == 300 + i
        assert imu_data[i].gyr_x == 50 + i
        assert imu_data[i].gyr_y == 100 + i
        assert imu_data[i].gyr_z == 150 + i
