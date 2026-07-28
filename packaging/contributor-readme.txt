llm-on-device-benchmark — contributor kit
=========================================

This folder measures how fast small LLMs run on YOUR machine and packs the
measurements into one small file you attach to a GitHub issue. Everything is
self-contained: no Python, compiler, or package manager needed, and nothing
is installed outside this folder (plus a per-user cache for Python itself).

How to run
----------
  Linux / macOS:  ./run.sh
  Windows:        double-click run.bat

That's it. The script tells you what it's doing at every step and ends by
printing the file to attach and the link to attach it at. It is safe to
interrupt and re-run — downloads resume, finished steps are skipped fast.

What to expect
--------------
- First run downloads models from Hugging Face (tens of GB — check you have
  the disk space and don't pay per GB).
- The benchmark itself typically takes 1–3 hours. Keep the machine plugged
  in and otherwise idle; results from a busy or battery-throttled machine
  are noisy.
- macOS will warn about an app from an unidentified developer. Either
  right-click → Open once, or run:  xattr -dr com.apple.quarantine .
- Windows SmartScreen may warn similarly — "More info" → "Run anyway".

What you share
--------------
The submission file (~300 KB) contains: timing and memory measurements, your
CPU/GPU/RAM model names and OS family, the library versions used, and a few
short model-generated text snippets so a reviewer can see the model ran
correctly. It does NOT contain your hostname, username, file paths, or
anything else about your machine or files. You can look inside before
attaching it: it's a plain .tar.gz of JSON.

Problems?
---------
Open an issue: https://github.com/vlntnplt/llm-on-device-benchmark/issues/new
and paste the output of the failing step (plus exe-error.log if it exists).
