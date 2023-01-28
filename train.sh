python train.py \
    --train_gen_data data/qqp_paragen_train.json \
    --dev_gen_data data/qqp_paragen_dev.json \
    --model_postfix bart_qqp_1gen \
    --generative

python train.py \
    --train_gen_data data/qqp_paragen_train.json \
    --dev_gen_data data/qqp_paragen_dev.json \
    --model_postfix bart_qqp_1gen_2gen+triecl \
    --from_checkpoint bart_qqp_1gen \
    --generative --contrastive \
    --log_interval 500

python train.py \
    --train_gen_data data/qqp_paragen_train.json \
    --dev_gen_data data/qqp_paragen_dev.json \
    --model_postfix bart_qqp_1gen+triecl \
    --generative --contrastive

python train.py \
    --train_gen_data data/qqp_paragen_train.json \
    --dev_gen_data data/qqp_paragen_dev.json \
    --model_postfix bart_qqp_1gen_2triecl \
    --from_checkpoint bart_qqp_1gen \
    --contrastive \
    --log_interval 500