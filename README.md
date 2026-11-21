# Run the speech page on your own computer

A page where you give any audio (a file or the microphone) and get, for each stretch of speech, the language and the transcript in that language's script. Everything runs on your computer. Nothing is sent anywhere, and no internet is needed after setup.

## What you need
- macOS, Linux, or Windows with WSL2 (Ubuntu)
- Python 3.10, 3.11 or 3.12
- About 3 GB of free disk and 8 GB of free RAM
- A C++ compiler for one package (`kenlm`): macOS `xcode-select --install`, Ubuntu `sudo apt install build-essential python3-dev`

This folder already contains the model (`results/checkpoints/xlsr53_s3e/best.pt`, 1.3 GB), the language model (`decoding/lm/mix.bin`, 1.8 GB), the dictionaries and the code.

## Set up once
```bash
cd universal_indic_asr_laptop
chmod +x setup.sh run.sh
./setup.sh
```
This creates a private Python environment in `venv/` and installs the packages. It takes a few minutes.

## Start the page
```bash
./run.sh
```
Wait about one to two minutes for the models to load. The terminal then prints a link like `http://127.0.0.1:8765/?t=...` and your browser opens it. Use the whole link: the part after `?t=` is a password that is made again each time you start.

On the page: drop an audio file, record from the microphone, or choose "Try an example". Press Transcribe. Click a block in the coloured bar to jump to that part of the audio.

Press Ctrl+C in the terminal to stop.

## Windows
Install WSL2 with Ubuntu (`wsl --install` in PowerShell), copy this folder inside the Ubuntu file system (for example `~/universal_indic_asr_laptop`, not `/mnt/c/...`, which is slow), and run the same commands in the Ubuntu terminal. Open the printed link in your normal Windows browser. If the browser does not open by itself, copy the link.

Native Windows is not supported because `kenlm` is hard to build there.

## Command line instead of the page
```bash
source venv/bin/activate
PYTHONPATH=. python transcribe.py recording.wav --device cpu --out result.json
```

## If something goes wrong
- `MISSING results/checkpoints/...`: the large files did not copy. Copy the whole folder again; `best.pt` must be about 1.3 GB and `mix.bin` about 1.8 GB.
- `kenlm` fails to install: install the compiler first (see above), then run `./setup.sh` again.
- The page says the token is wrong: use the link printed in the terminal this time, not an old one.
- The microphone button says it needs a secure page: open the page through `127.0.0.1` or `localhost`, as the printed link does.
- Slow: on CPU, expect roughly 1 to 2 times the length of the clip (a 5 s clip took about 10 s on an 8-core machine in our test). A GPU is much faster. The first request after starting can take longer.

## What to expect
Real speech works better than synthetic voices. On our synthetic test clips the exact language sequence came out right on about half. A language the model never trained on (for example Odia) is usually labelled Bengali. The example list includes one such case on purpose.
