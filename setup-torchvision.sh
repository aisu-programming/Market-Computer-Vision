if ! [ -s torchvision-0.10.0a0-cp37-cp37m-linux_armv7l.whl ]
then
    wget https://github.com/aisu-programming/Market-Computer-Vision/releases/download/v1.0/torchvision-0.10.0a0-cp37-cp37m-linux_armv7l.whl
fi
pip3 install torchvision-0.10.0a0-cp37-cp37m-linux_armv7l.whl
rm torchvision-0.10.0a0-cp37-cp37m-linux_armv7l.whl