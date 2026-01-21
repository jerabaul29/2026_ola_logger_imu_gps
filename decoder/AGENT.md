# Python decoder for the raw data files

## Overview

- the goal is to decode the raw data files generated from the OLA logger using python code
- see the spec in the `./SPEC.md` file; make sure to read it
- the OLA (OpenLogArtemis) logger runs a C/C++ bare metal firmware are logs to SD card
- the OLA logger logs an IMU (ISM330DHCX) and a GNSS (SAM-M10Q GPS, both logging some navigation data and the PPS 1Hz rising edge)

## Code and data organization

- the `./decoder.py` module contains all the utilities to easily decode a single data file at a time
- the file `./DATA_BOOT_0055_TIME_20260120T211500.dat` contains an example of data file
- the `./example_decode.py` script contains an example of how to decode the data file using the decoder functions
- the `./test_decoder.py` module contains the tests to run
- write and keep up to date a README.md file to explain how to install the mamba env, run the code, etc.

## Code practices

Everything for parsing etc is written in python

- have tests to run with pytest
- have logging with loguru
- write simple, idiomatic python code
- use typing / type annotation for the functions
- use formal checkers for the code (install these in the mamba environment):
  - check spell in the code with `pylint`
  - check and lint the code with `ruff` + `pylint` + `flake8`
  - check code complexity with `complexipy`
  - generally, double check your code for good style, ease of understanding, good API / interface, safety and speed
- run test after each code change
- use established packages:
  - struct and struct.unpack for raw data handling (reading int16_t, long unsigned int, etc; long unsigned int is 32 bits)
  - dataclasses to model C/C++ structs
  - numpy for the arrays and dump the data as npy; use the type "object" for the arrays to contain the dataclasses objects; enrich the dataclass with "scaled" values for the IMU (float scaled values by the sensors sensitivity in addition to raw int16_t readings)
  - no more package imports needed

## Code setup and environment management: mamba

- use mamba; it should be pre installed on the system; do not install it yourself, do not add channels; fail if no mamba and ask user help
- use an environment.yml to define the necessary packages
- use, or create if necessary, a dedicated environment for working on this, with name: "ola_ism330dhcx_samm10q_decoder"; use only this env for running code here
- use only conda-forge channel (this should already be set by the mamba install available)

## Code architecture and conventions for the decoder

- magic constants should be ALL_CAPS_CONSTANTS defined at the start to make it easy to edit
- avoid object oriented if not necessary, make the code based on simple functions
- pass the ALL_CAPS_CONSTANTS as default args, it is ok to have many default args to the functions
- use individual functions to parse each raw data kind (i.e. parse 1 line of raw data)
- write a single function to parse a single file and write the npy to disk
- when dumping to disk, there should be an individual .npy file for each datakind, containing a numpy array of dataclass objects matching the object kind

## Misc

- at the end of each update to the code, go through the files and check that the documentation and readme are up to date
- when writing code, check that all guidelines present here are followed
- at the end of each code update, do a small review of the code, look for good possibilities for refactoring and improvement, and if there are clear improvement possibilities, work on implementing them
- at the end of each code update, make sure everything works - tests, spell check, linting, etc

