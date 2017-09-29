virtualenv --system-site-packages -p python3 ~/tensorflow_cpu
source ~/tensorflow_cpu/bin/activate
pip install --upgrade tensorflow
#virtualenv --system-site-packages -p python3 ~/tensorflow_gpu
#source ~/tensorflow_gpu/bin/activate
#pip install --upgrade tensorflow_gpu
pip install pandas
pip install sklearn
pip install scipy
pip install matplotlib
deactivate
