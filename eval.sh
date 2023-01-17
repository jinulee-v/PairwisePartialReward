export TOKENIZERS_PARALLELISM=false

python inference.py \
    --test_data data/qqp_paragen_test.json \
    --num_beams 24 \
    --num_beam_groups 24 \
    --diversity_penalty 0.5 \
    --model_store_path checkpoints \
    --model_postfix qqp_Gen

python eval.py \
    --model_store_path checkpoints \
    --model_postfix qqp_Gen