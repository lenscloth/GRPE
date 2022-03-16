# We found that underfitted pretrained model is better for HIV dataset.
# You need an A100 with 80GiB for learning the command.
python gpp.py --dataset hiv  --batch-size 72 --world-size 1 --load pretrained_weight/pcqm4m_pretrained_standard_underfit.pt \
            --node-dim 768 --ffn-dim 768 --nhead 32 --num-layer 12 \
            --attention-dropout 0.0 --dropout 0.0 --peak-lr 2e-4 \
            --warmup-epoch 3 --max-epoch 100 --early-stop-epoch 5 --save result/hiv_standard