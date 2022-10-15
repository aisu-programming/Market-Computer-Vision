if ! [ -s torch-1.9.0a0-cp37-cp37m-linux_armv7l.whl ]
then
    wget https://github.com/aisu-programming/Market-Computer-Vision/releases/download/v1.0/torch-1.9.0a0-cp37-cp37m-linux_armv7l.whl
fi
pip3 install torch-1.9.0a0-cp37-cp37m-linux_armv7l.whl
rm torch-1.9.0a0-cp37-cp37m-linux_armv7l.whl