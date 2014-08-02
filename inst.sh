#! /bin/bash
##################################################################################
### Add the directory you want to install to. eg 
install_dir=/home/gurpreet/sft/python/env1
#install_dir=

yes | pip uninstall pmfcalculator
python setup.py clean --all
python setup.py install --prefix $install_dir
