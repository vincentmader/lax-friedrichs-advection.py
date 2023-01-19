#!/bin/sh

#   Create virtual environment for python, if not already done.
    [ -d ../.venv ] || python3 -m virtualenv ../.venv

#   Install python dependencies.
    ../.venv/bin/pip3 install -r ../requirements.txt

#   Pull matplotlib stylesheets from GitHub.
    if [ -d ../src/mpl-styles ]; then
        cd ../src/mpl-styles/ && git pull && cd ../../
    else
        git clone https://github.com/vincentmader/mpl-styles ../src/mpl-styles
    fi
