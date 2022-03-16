# PCBA Standard Model
python gpp.py --dataset pcba --world-size 4 --load pretrained_weight/pcqm4m_pretrained_standard.pt \
            --node-dim 768 --ffn-dim 768 --nhead 32 --max-hop 5 --num-layer 12 \
            --attention-dropout 0.3 --dropout 0.3 --peak-lr 3e-4 \
            --warmup-epoch 3 --max-epoch 200 --save result/pcba_standard

# PCBA Large Model
python gpp.py --dataset pcba --world-size 4 --load pretrained_weight/pcqm4m_pretrained_large.pt \
            --node-dim 1024 --ffn-dim 1024 --nhead 32 --max-hop 5 --num-layer 18 \
            --attention-dropout 0.3 --dropout 0.3 --peak-lr 3e-4 \
            --warmup-epoch 3 --max-epoch 200 --save result/pcba_large
