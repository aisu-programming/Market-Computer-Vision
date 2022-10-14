sudo apt-get update -y
sudo apt-get install build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev -y
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz
tar zxf Python-3.7.9.tgz
cd Python-3.7.9
./configure
make -j 4
sudo make altinstall
cd ..
sudo rm -r Python-3.7.4
rm Python-3.7.4.tar.xz

wget https://github.com/aisu-programming/Market-Computer-Vision/releases/download/Customer-Flow-Detection/torch-1.9.0a0-cp37-cp37m-linux_armv7l.whl
pip install torch-1.9.0a0-cp37-cp37m-linux_armv7l.whl

wget https://github.com/aisu-programming/Market-Computer-Vision/releases/download/Customer-Flow-Detection/torchvision-0.10.0a0-cp37-cp37m-linux_armv7l.whl
pip install torchvision-0.10.0a0-cp37-cp37m-linux_armv7l.whl

# wget https://github.com/aisu-programming/Market-Computer-Vision/releases/download/Vegetable-Instance-Segmentation/tensorflow-2.4.0rc4-cp37-none-linux_armv7l.whl
# pip install tensorflow-2.4.0rc4-cp37-none-linux_armv7l.whl

# wget https://www.piwheels.org/simple/h5py/h5py-2.10.0-cp37-cp37m-linux_armv7l.whl
# pip install h5py-2.10.0-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/numpy/numpy-1.19.5-cp37-cp37m-linux_armv7l.whl
# pip install numpy-1.19.5-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/grpcio/grpcio-1.32.0-cp37-cp37m-linux_armv7l.whl
# pip install grpcio-1.32.0-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/opencv-python/opencv_python-4.5.3.56-cp37-cp37m-linux_armv7l.whl
# pip install opencv_python-4.5.3.56-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/scikit-image/scikit_image-0.18.3-cp37-cp37m-linux_armv7l.whl
# pip install scikit_image-0.18.3-cp37-cp37m-linux_armv7l.whl

# wget https://www.piwheels.org/simple/matplotlib/matplotlib-3.4.3-cp37-cp37m-linux_armv7l.whl
# pip install matplotlib-3.4.3-cp37-cp37m-linux_armv7l.whl

# pip install pycocotools
# pip install tqdm