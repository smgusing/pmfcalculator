===============================================
pmfcalculator: A package to compute PMF profiles from biased simulations
===============================================

:Author: Gurpreet Singh and Alex Chen
:Contact: togurpreet@gmail.com
:License: Read LICENSE.txt 

-----------------------------------------------
Introduction
----------------------------------------------- 
pmfcalculator is a python module that can be used for computing  N dimensional potential of mean force profiles
 from the biased simulations.
Currently two methods are implimented
- Traditional WHAM
- ZHU method

-----------------------------------------------
Installation
-----------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Prerequists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Numpy (developed using version 1.8.0)
- PyYaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quick install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
python setup.py install


The installation script ``inst.sh`` is provided. Please modify it according to your environment.
The script will **not** work as is.

.. DANGER:: The script naively tries to remove the previous installation using ``rm`` command, **DO NOT USE IT WITHOUT MODIFICATIONS**.  

-------------------------------------------------
Usage
-------------------------------------------------
Use ``calculatepmf.py -h`` for further help.
   





