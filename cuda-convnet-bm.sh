export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

python convnet.py --data-path /home/pziewiec/mnist_normalized --save-path /home/pziewiec/tmp --train-range 1-6 --test-range 7 --epochs 100 --layer-def /home/pziewiec/cuda-convnet2/layers/layer-MNIST.cfg --layer-params /home/pziewiec/cuda-convnet2/layers/layer-params-MNIST.cfg --data-provider MNIST --gpu 0 --mini 128 --test-freq 10 >&1 | tee cuda-convnet_out.txt
