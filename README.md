# Market-Computer-Vision
To clone this repository, type `git clone --recursive https://github.com/aisu-programming/Market-Computer-Vision.git`.

Notice that the option `--recursive` is necessary because of submodules.

## Download models
Type the command `sh download-weights.sh` to download all model's weights files.

## Environment setup
1. Type the command `sh setup-torch.sh` to install PyTorch (version 1.9) for Raspberry Pi 32-bits OS with Python 3.7.
2. Type the command `sh setup-torchvision.sh` to install torchvision (version 0.10) for Raspberry Pi 32-bits OS with Python 3.7.
3. Type the command `sh setup-others.sh` to install all other requirements.
4. Setup the Google Sheet API. See [tutorial](https://github.com/aisu-programming/Market-Computer-Vision/blob/main/GOOGLE_SHEET_API_TUTORIAL.md).

## Usage
The main command is `python3 detect.py`.

There are serveral options can be add:

- `--fd`: enable the *Footfall-Detection* function. (default: *False*)
  > Example: `python3 detect.py --fd`.
- `--fd-img-size <int>`: the size of input images for the *Footfall-Detection* task will be resize to this. (default: *64*)
  > Example: `python3 detect.py --fd-img-size 32` will set the size of input images to 32x32.

<!--
- `--vis-full-thres`: if the prediction amount of input is more than this value, it will be classify to "Full" status. (default: _70_)
- `--vis-less-thres`: if the prediction amount of input is more than this value, it will be classify to "Less" status. (default: _30_)
  > Example: `python3 detect.py --vis-full-thres 60 --vis-less-thres 40` will view a 50% amount prediction as "Less" status.
- `--vis-smoothing-len`: the length of the smoothing array which to prevent unstable predictions. (default: _10_)
-->

- `--sed`: enable the *Shelf-Empty-Detection* function. (default: *False*)
  > Example: `python3 detect.py --sed`.
- `--sed-img-size <int>`: the size of input images for the *Shelf-Empty-Detection* task will be resize to this. (default: *256*)
  > Example: `python3 detect.py --sed-img-size 128` will set the size of input images to 128x128.
- `--sed-alert-amount-thres <float>`: the amount threshold to view a stock as empty. (default: *0.5*)
  > Example: `python3 detect.py --sed-alert-amount-thres 0.3` will set the alert amount to 0.3.
- `--sed-smoothing-len <int>`: the length of the smoothing array which to prevent unstable predictions. (default: _10_)
  > Example: `python3 detect.py --sed-smoothing-len 20` will set the length of the smoothing array to 20.
- `--test`: use the testing images in *test* directory to test the inference. (default: _False_)
  > Example: `python3 detect.py --test`.
- `--view-img`: show real-time camera images. (default: _False_)
  > Example: `python3 detect.py --view-img`.
- `--sleep <int>`: the sleep time between every inference. (default: _0_)
  > Example: `python3 detect.py --sleep 3` for sleeping 3 seconds after every image.
- `--save-img`: save inferenced images to the output directory or not. (default: _False_)
  > Example: `python3 detect.py --save-img`.
- `--save-img-interval <int>`: save inferenced images to output directory per seconds. (default: _900_)
  > Example: `python3 detect.py --save-img-interval 60` for saving image every 1 minute (60 seconds).
- `--save-csv`: save results to csv file or not. (default: _False_)
  > Example: `python3 detect.py --save-csv`.
- `--save-csv-interval <int>`: save results to csv file per seconds. (default: _900_)
  > Example: `python3 detect.py --save-csv-interval 60` for saving records every 1 minute (60 seconds).
- `--save-google-sheet`: save results to google sheet or not. (default: _False_)
  > Example: `python3 detect.py --save-google-sheet`.
- `--save-google-sheet-interval <int>`: save results to google sheet per seconds. (default: *900*)
  > Example: `python3 detect.py --save-google-sheet-interval 60` for saving records every 1 minute (60 seconds).

Combine above options or edit default value of each options in *detect.py*.

## Some data
|                                                      | Raspberry Pi 3B         | Raspberry Pi 4B         |
| ---------------------------------------------------- | ----------------------- | ----------------------- |
| Installation time (setup-*.sh + download-weights.sh) | 31 minutes              | 23 minutes              |
| Footfall-Detection speed (input img size = 64)       | 2~4 sec / per inference | 1.4 sec / per inference |
| Shelf-Empty-Detection speed (input img size = 256)   | 6~8 sec / per inference | 2~3 sec / per inference |

## About the OS
I used the [Raspberry Pi Imager](https://www.raspberrypi.com/software/).

The OS I chose is the **Raspberry Pi OS (Legacy)**, which is 32-bits and with the Python version 3.7.
