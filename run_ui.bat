@echo off
cd /d %~dp0
python -m streamlit run ui.py
pause