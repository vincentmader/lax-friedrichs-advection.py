#!/bin/sh

[ -d ../.venv ] || python3 -m virtualenv ../.venv
../.venv/bin/pip3 install -r ../requirements.txt
