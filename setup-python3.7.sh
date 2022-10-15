# sudo apt-get update -y
# sudo apt-get install build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev -y
# wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz
# tar xf Python-3.7.9.tar.xz
# cd Python-3.7.9
# ./configure
# make -j 4
# sudo make altinstall
# cd ..
# sudo rm -r Python-3.7.9
# rm Python-3.7.9.tar.xz
# sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.7 1
# sudo update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.7 1
# sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.7 1
# sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.7 1
# sudo mv /usr/bin/lsb_release /usr/bin/lsb_release_bak1223