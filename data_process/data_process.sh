#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m $(dirname "$0")/csv2json.py
$PYTHON -m $(dirname "$0")/crop_patch_test_mutilprocess.py
$PYTHON -m $(dirname "$0")/generate_test.py
$PYTHON -m $(dirname "$0")/generate_raw_test.py
