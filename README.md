# Market-Computer-Vision

## Environment setup
All the requiring packages are list in _requirements.txt_.

Simply type command `pip3 install -r requirements.txt` to install them.

However, download source codes and build it on _RaspberryPi 3_ is extremely slow. (It may cost more than 3 days.)

Also, due to slow WiFi speed of _RaspberryPi 3_, the connection to download packages from the cloud is likely to be cut off because it cost too much time.

Accordingly, use `wget` to download packages first then install will be a proper way.

To do that, type the command `sh setup.sh`, `bash setup.sh` or just `setup.sh`.

Detailed commands are all in the file, follow those commands if something happened accidentally and interrupted the process.

## Usage
The main command is `python3 detect.py`.

There are serveral options to adjust:
- cfd-img-size: the size of input images for the _Customer-Flow-Detection_ task will be resize to this. (default: _96_)
  > Example: `python3 detect.py --cfd-img-size 32` will set the size of input images to 32x32.
- view-img: add this option to show real-time camera images. (default: _False_)
  > Example: `python3 detect.py --view-img`.
- sleep: the sleep time after every image was inferenced and predicted. (default: _1_)
  > Example: `python3 detect.py --sleep 3` for sleeping 3 seconds after every image.
- save-img: save image to the output directory or not. (default: _False_)
  > Example: `python3 detect.py --save-img`.
- save-img-interval: save image to output directory per seconds. (default: _10_)
  > Example: `python3 detect.py --save-img-interval 60` for saving image every 1 minute.
- save-csv: save every inference results and prediction results to csv file or not. (default: _False_)
  > Example: `python3 detect.py --save-csv`.
- save-csv-interval: save records to csv file per seconds. (default: _5_)
  > Example: `python3 detect.py --save-csv-interval 60` for saving records every 1 minute.

Please combine options or edit default value of each options in _detect.py_.
