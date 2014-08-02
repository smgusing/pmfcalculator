#! /bin/bash

source ~/.bashrc

## set your python environment
lpy
##


function gen_1d_test_data {
rm -rf test1d
mkdir test1d 
cp gen_1d_testdata.py smooth.py plot_1d_results.py test1d/
cd test1d ; ./gen_1d_testdata.py ; cd ..
}



function test_1d_wham {

cd test1d

calc1dpmf.py -projfile dummy.yaml -maxiter 10000  \
-nbins 100 -tolerance 1e-7 -chkdur 500 -temperature 300 \
-range 0.0 7.95 -nbootstrap 4 -zerofe 7 -l info 

if  [[ -f  "pmf1d3.txt" ]]
then
    ./plot_1d_results.py
    
    echo "Results should be plotted in 1dpmf.png"
else
    
    echo " Could not find the results"
fi

cd ..

}




function test_1d_zhu {


cd test1d

calc1dpmf_Zhu.py -projfile dummy.yaml -maxiter 10000  \
-nbins 100 -tolerance 1e-7 -chkdur 500 -temperature 300 \
-range 0.0 7.95 -nbootstrap 4 -zerofe 7 -l debug 

if  [[ -f  "pmf1d3.txt" ]]
then
    ./plot_1d_results.py
    
    echo "Results should be plotted in 1dpmf.png"
else
    
    echo " Could not find the results"
fi

cd ..

}


function gen_2d_test_data {
rm -rf test2d
mkdir test2d 
cp gen_2d_testdata.py smooth.py plot_2d_results.py test2d/
cd test2d ; ./gen_2d_testdata.py ; cd ..





}



function test_2d_wham {
    
    cd test2d
    
    calc2dpmf.py -projfile dummy.yaml -maxiter 1000  \
    -nbins 100 45 -tolerance 1e-3 -chkdur 50 -temperature 298.15 \
    -range -0.0 7.95 0.0 180.0 -nbootstrap 1 -bootbegin 0 -zerofe 7.5 90.0 -l debug 

    if  [[ -f  "pmf2d1.npz" ]]
    then
        ./plot_2d_results.py
        
        echo "Results should be plotted in pmf2d.png"
    else
        
        echo " Could not find the results"
    fi

    cd ..

}




# 
# function test1dmbar {
# 
# cd results1d
# calc1dpmf_mbar.py -projfile ../test1d.yaml -maxiter 10000 -fsuffix dh \
# -nbins 100 -tolerance 1e-4 -chkdur 500 -temperature 300 \
# -range 0.0 7.95 -nbootstrap 4 
# 
# }

# 
# 
# 
# function test2dmbar {
# cd results2d
# calc2dpmf_mbar.py -projfile ../a2d.yaml -maxiter 1000 -fsuffix dh \
# -nbins 100 45 -tolerance 1e-3 -chkdur 50 -temperature 298.15 \
# -range -0.0 7.95 0.0 180.0 -nbootstrap 5  
# }
# 
# function test2dhummer {
# cd results2d
# rm *.npz .fe.npz
# calc2dpmf_zhu.py -projfile ../a2d.yaml -maxiter 1000 -fsuffix dh \
# -nbins 100 45 -tolerance 1e-3 -chkdur 50 -temperature 298.15 \
# -range -0.0 7.95 0.0 180.0 -nbootstrap 5  
# }


function wham1d {

## Generate dummy data for 1D 
gen_1d_test_data

## test 1D wham
test_1d_wham

}

function zhu1d {

## Generate dummy data for 1D 
gen_1d_test_data

## test 1D wham
test_1d_wham

}




#test1d_all

case "$1" in
    gen_1d_test_data)
        gen_1d_test_data
        ;;

    test_1d_wham)
        test_1d_wham
        ;;

    gen_2d_test_data)
        gen_2d_test_data
        ;;
        
    test_2d_wham)
        test_2d_wham
        ;;
        
    test_1d_zhu)
        test_1d_zhu
        ;;
        
        
    *)
        echo $"usage: $0:{ gen_1d_test_data| test_1d_wham | gen_2d_test_data | test_2d_wham}"
        exit 1
esac
    
    


#test1dmbar
#test1dhummer
#test2d
#test2dmbar
#test2dhummer
#test2d
#test2dmbar
