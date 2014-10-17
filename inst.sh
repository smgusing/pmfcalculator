#! /bin/bash
##################################################################################
### Add the directory you want to install to. eg 
install_dir=/home/gurpreet/sft/python/env1
#install_dir=
function inst {
	yes | pip uninstall pmfcalculator
	python setup.py clean --all
	python setup.py install --prefix $install_dir
}

function doc {
# In order to build the documentation the path to the installed package $pkg_path should be set
pkg_path=${install_dir}/lib/python2.7/site-packages/pmfcalculator
sphinx-apidoc $pkg_path -o doc --full
cd doc; make html

}

function clean {
	python setup.py clean --all
	rm -rvf doc/_*
  
}


case $1 in
	clean)
	clean
	;;

	doc)
	doc
	;;

	inst)
	inst
	;;

	*)
	echo $"usage: $0 {clean|doc|inst}"
esac

