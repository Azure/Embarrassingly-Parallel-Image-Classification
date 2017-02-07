#!/bin/bash
#
# This install script generously shared by Miruna Oprescu
# (then lightly modified by Mary Wahl), Microsoft Corporation, 2017

cntk_home="/usr/hdp/current"
cd $cntk_home
curl "BinaryDrop/CNTK-2-0-beta10-0-Linux-64bit-CPU-Only.tar.gz" | tar xzf -
cd ./cntk/Scripts/install/linux 
sed -i "s#"ANACONDA_PREFIX=\"\$HOME/anaconda3\""#"ANACONDA_PREFIX=\"\/usr/bin/anaconda\""#g" install-cntk.sh
sed -i "s#"\$HOME/anaconda3"#"\$ANACONDA_PREFIX"#g" install-cntk.sh
./install-cntk.sh --py-version 35

sudo /usr/bin/anaconda/envs/cntk-py35/bin/pip install pillow
sudo /usr/bin/anaconda/envs/cntk-py35/bin/pip install tensorflow

sudo mkdir /tmp/resnet
sudo chmod -R 777 /tmp/resnet