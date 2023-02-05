# python train.py \
#     --train_gen_data data/qqp_paragen_train.json \
#     --dev_gen_data data/qqp_paragen_dev.json \
#     --model_postfix bart_qqp_1gen \
#     --generative

python train.py \
    --train_gen_data data/qqp_paragen_train.json \
    --dev_gen_data data/qqp_paragen_dev.json \
    --model_postfix bart_qqp_1gen_2gen+brio \
    --from_checkpoint bart_qqp_1gen \
    --generative --contrastive \
    --log_interval 500

python train.py \
    --train_gen_data data/qqp_paragen_train.json \
    --dev_gen_data data/qqp_paragen_dev.json \
    --model_postfix bart_qqp_1gen+brio \
    --generative --contrastive

# python train.py \
#     --train_gen_data data/parabank_train.json \
#     --dev_gen_data data/parabank_dev.json \
#     --model_postfix bart_parabank_1gen \
#     --generative \
#     --log_interval 50000

# python train.py \
#     --train_gen_data data/parabank_train.json \
#     --dev_gen_data data/parabank_dev.json \
#     --model_postfix bart_parabank_1gen_2gen+brio \
#     --from_checkpoint bart_parabank_1gen \
#     --generative --contrastive \
#     --log_interval 10000

# python train.py \
#     --train_gen_data data/parabank_train.json \
#     --dev_gen_data data/parabank_dev.json \
#     --model_postfix bart_parabank_1gen+brio \
#     --generative --contrastive\
#     --log_interval 50000