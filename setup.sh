wget https://github.com/aisu-programming/Market-Computer-Vision/releases/download/v1.0/torch-1.9.0a0-cp37-cp37m-linux_armv7l.whl
pip3 install torch-1.9.0a0-cp37-cp37m-linux_armv7l.whl
wget https://github.com/aisu-programming/Market-Computer-Vision/releases/download/v1.0/torchvision-0.10.0a0-cp37-cp37m-linux_armv7l.whl
pip3 install torchvision-0.10.0a0-cp37-cp37m-linux_armv7l.whl

# wget https://github.com/aisu-programming/Market-Computer-Vision/releases/download/Vegetable-Instance-Segmentation/tensorflow-2.4.0rc4-cp37-none-linux_armv7l.whl
# pip3 install tensorflow-2.4.0rc4-cp37-none-linux_armv7l.whl
# wget https://www.piwheels.org/simple/h5py/h5py-2.10.0-cp37-cp37m-linux_armv7l.whl
# pip3 install h5py-2.10.0-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/numpy/numpy-1.19.5-cp37-cp37m-linux_armv7l.whl
# pip install numpy-1.19.5-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/grpcio/grpcio-1.32.0-cp37-cp37m-linux_armv7l.whl
# pip3 install grpcio-1.32.0-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/opencv-python/opencv_python-4.5.3.56-cp37-cp37m-linux_armv7l.whl
# pip3 install opencv_python-4.5.3.56-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/scikit-image/scikit_image-0.18.3-cp37-cp37m-linux_armv7l.whl
# pip3 install scikit_image-0.18.3-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/matplotlib/matplotlib-3.4.3-cp37-cp37m-linux_armv7l.whl
# pip3 install matplotlib-3.4.3-cp37-cp37m-linux_armv7l.whl

pip3 install -r requirements.txt

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libhdf5-dev
sudo apt-get install -y libhdf5-serial-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libjasper-dev
sudo apt-get install -y libqtgui4
sudo apt-get install -y libqt4-test
sudo apt-get install -y libilmbase-dev
sudo apt-get install -y libopenexr-dev
sudo apt-get install -y libopenblas-dev
sudo apt-get install -y libgstreamer1.0-dev
sudo apt-get install -y libavcodec-dev
sudo apt-get install -y libavformat-dev
sudo apt-get install -y libswscale-dev
sudo apt-get install -y libwebp-dev
