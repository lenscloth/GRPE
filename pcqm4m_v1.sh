# PCQM4M Standard Model
python gpp.py --dataset lsc-v1 --world-size 8 --batch-size 128 \
              --node-dim 768 --ffn-dim 768 --nhead 32 --num-layer 12 \
              --warmup-epoch 3 --max-epoch 400 --attention-dropout 0.1 --dropout 0.1 \
              --weight-decay 0.0 --peak-lr 2e-4 --max-hop 5 \
              --valid-after 200 --valid-every 10 --save result/lsc_v1_large_model

# Evaluate Pretrained PCQM4M Standard Model
python gpp.py --dataset lsc-v1 --world-size 8 --batch-size 128 \
              --node-dim 768 --ffn-dim 768 --nhead 32 --num-layer 12 \
              --warmup-epoch 3 --max-epoch 400 --attention-dropout 0.1 --dropout 0.1 \
              --weight-decay 0.0 --peak-lr 2e-4 --max-hop 5 \
              --valid-after 200 --valid-every 10 --save result/lsc_v1_large_model_retrain \
              --load pretrained_weight/pcqm4m_pretrained_standard.pt --load-all
