# CIFAR10 Tiny
python gpp.py --dataset cifar10 --num-node-type -5 --batch-size 64 \
              --max-hop 3 --num-layer 4 --node-dim 64 --ffn-dim 64 --nhead 8 \
              --max-epoch 300 --save results/cifar10_tiny 
    
# CIFAR10 Tiny2x
python gpp.py --dataset cifar10 --num-node-type -5 --batch-size 64 \
              --max-hop 3 --num-layer 4 --node-dim 128 --ffn-dim 128 --nhead 16 \
              --max-epoch 300 --save results/cifar10_tiny2x 
