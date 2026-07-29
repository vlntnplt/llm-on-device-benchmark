@echo off
rem Double-clickable wrapper: runs the real entry point (run.ps1) and keeps the
rem window open so the final instructions stay readable.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
pause
