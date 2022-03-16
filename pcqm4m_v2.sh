# PCQM4Mv2 Standard Model
python gpp.py --dataset lsc-v2 --world-size 8 --batch-size 128 \
              --node-dim 768 --ffn-dim 768 --nhead 32 --num-layer 12 --max-hop 5 \
              --warmup-epoch 3 --max-epoch 400 --attention-dropout 0.1 --dropout 0.1 \
              --weight-decay 0.0 --peak-lr 2e-4 --max-hop 5 \
              --valid-after 200 --valid-every 10 --save result/lsc_v2_standard

# PCQM4Mv2 Large Model
python gpp.py --dataset lsc-v2 --world-size 8 --batch-size 64 \
              --node-dim 1024 --ffn-dim 1024 --nhead 32 --num-layer 18 --max-hop 5 \
              --warmup-epoch 3 --max-epoch 400 --attention-dropout 0.1 --dropout 0.1 \
              --weight-decay 0.0 --peak-lr 2e-4 --max-hop 5 \
              --valid-after 200 --valid-every 10 --save result/lsc_v2_large

# Evalute Pretrianed PCQM4Mv2 Standard Model
python gpp.py --dataset lsc-v2 --world-size 8 --batch-size 128 \
              --node-dim 768 --ffn-dim 768 --nhead 32 --num-layer 12 --max-hop 5 \
              --warmup-epoch 3 --max-epoch 400 --attention-dropout 0.1 --dropout 0.1 \
              --weight-decay 0.0 --peak-lr 2e-4 --max-hop 5 \
              --valid-after 200 --valid-every 10 --save result/lsc_v2_standard_retrain \
              --load pretrained_weight/pcqm4mv2_pretrained_standard.pt --load-all
