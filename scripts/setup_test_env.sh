#!/usr/bin/env bash
# Install dependencies required for running the test suite.
set -e

# Install CPU version of PyTorch and Lightning
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.0
pip install lightning==2.1.2

# Install the rest of the dependencies
pip install -r requirements.txt

