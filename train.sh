# python train_qqp.py \
#     --train_gen_data data/qqp_paragen_train.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --model_store_path checkpoints \
#     --epoch 10 \
#     --torch_seed 0 \
#     --model_postfix qqp_Gen

python train_qqp.py \
    --train_gen_data data/qqp_paragen_train.json \
    --dev_gen_data data/qqp_paragen_dev.json \
    --model_store_path checkpoints \
    --epoch 10 \
    --torch_seed 0 \
    --fine_tune \
    --from_checkpoint qqp_Gen \
    --model_postfix qqp_Contrastive