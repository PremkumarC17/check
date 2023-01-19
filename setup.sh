#!/bin/bash

apt-get update
apt-get install -y python3-pip
pip install --upgrade pip
pip3 install -r requirements.txt
streamlit run app.py