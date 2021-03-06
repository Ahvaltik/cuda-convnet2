export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

python convnet.py --data-path /home/pziewiec/hpdb_normalized --save-path /home/pziewiec/tmp/biwi --train-range 0-12 --test-range 14 --epochs 100 --layer-def /home/pziewiec/cuda-convnet2/layers/layer-BIWI.cfg --layer-params /home/pziewiec/cuda-convnet2/layers/layer-params-BIWI.cfg --data-provider BIWI --gpu 0 --mini 1024 --test-freq 10 >&1 | tee cuda-convnet_test_out.txt
