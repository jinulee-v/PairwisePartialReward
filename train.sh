# python train_qqp.py \
#     --train_gen_data data/qqp_paragen_train.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --train_ident_data data/qqp_paraident_train.json \
#     --dev_ident_data data/qqp_paraident_dev.json \
#     --model_store_path checkpoints \
#     --generation_loss_weight 0.5 \
#     --bc_ordering_loss_weight 0.5 \
#     --epoch 5 \
#     --torch_seed 0 \
#     --model_postfix qqp_Gen_BCOrder

# python train_qqp.py \
#     --train_gen_data data/qqp_paragen_train.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --train_ident_data data/qqp_paraident_train.json \
#     --dev_ident_data data/qqp_paraident_dev.json \
#     --from_checkpoint qqp_Gen+Ident \
#     --model_store_path checkpoints \
#     --generation_loss_weight 0.5 \
#     --bc_classification_loss_weight 0.5 \
#     --epoch 5 \
#     --torch_seed 0 \
#     --model_postfix qqp_Gen_BCClass_pretrained

# python train_qqp.py \
#     --train_gen_data data/qqp_paragen_train_8000.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --train_ident_data data/qqp_paraident_train.json \
#     --dev_ident_data data/qqp_paraident_dev.json \
#     --model_store_path checkpoints \
#     --generation_loss_weight 0.5 \
#     --bc_distance_loss_weight 0.5 \
#     --epoch 8 \
#     --torch_seed 0 \
#     --model_postfix qqp_Gen_BCDist_8000

# python train_qqp.py \
#     --train_gen_data data/qqp_paragen_train_8000.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --train_ident_data data/qqp_paraident_train.json \
#     --dev_ident_data data/qqp_paraident_dev.json \
#     --model_store_path checkpoints \
#     --generation_loss_weight 1 \
#     --epoch 8 \
#     --torch_seed 0 \
#     --model_postfix qqp_Gen_8000


# python train_qqp.py \
#     --train_gen_data data/qqp_paragen_train_20000.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --train_ident_data data/qqp_paraident_train.json \
#     --dev_ident_data data/qqp_paraident_dev.json \
#     --model_store_path checkpoints \
#     --generation_loss_weight 0.5 \
#     --bc_distance_loss_weight 0.5 \
#     --epoch 8 \
#     --torch_seed 0 \
#     --model_postfix qqp_Gen_BCDist_20000

# python train_qqp.py \
#     --train_gen_data data/qqp_paragen_train_20000.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --train_ident_data data/qqp_paraident_train.json \
#     --dev_ident_data data/qqp_paraident_dev.json \
#     --model_store_path checkpoints \
#     --generation_loss_weight 1 \
#     --epoch 8 \
#     --torch_seed 0 \
#     --model_postfix qqp_Gen_20000


# python train_qqp.py \
#     --train_gen_data data/qqp_paragen_train_4000.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --train_ident_data data/qqp_paraident_train.json \
#     --dev_ident_data data/qqp_paraident_dev.json \
#     --model_store_path checkpoints \
#     --generation_loss_weight 0.5 \
#     --bc_distance_loss_weight 0.5 \
#     --epoch 10 \
#     --torch_seed 0 \
#     --model_postfix qqp_Gen_BCDist_4000

python train_qqp.py \
    --train_gen_data data/qqp_paragen_train_4000.json \
    --dev_gen_data data/qqp_paragen_dev.json \
    --train_ident_data data/qqp_paraident_train.json \
    --dev_ident_data data/qqp_paraident_dev.json \
    --model_store_path checkpoints \
    --generation_loss_weight 1 \
    --epoch 10 \
    --torch_seed 0 \
    --model_postfix qqp_Gen_4000