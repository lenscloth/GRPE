# MNIST Tiny
python gpp.py --dataset mnist --num-node-type -3 --batch-size 64 \
              --max-hop 3 --num-layer 4 --node-dim 64 --ffn-dim 64 --nhead 8 \
              --max-epoch 300 --save results/mnist_tiny
    
# MNIST Tiny2x
python gpp.py --dataset mnist --num-node-type -3 --batch-size 64 \
              --max-hop 3 --num-layer 4 --node-dim 128 --ffn-dim 128 --nhead 16 \
              --max-epoch 300 --save results/mnist_tiny2x