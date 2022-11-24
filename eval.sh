export TOKENIZERS_PARALLELISM=false

echo "/////////////////////////////////////////////////////////"

python eval.py \
    --test_data data/qqp_paragen_test.json \
    --num_beam_groups 12 \
    --model_store_path checkpoints \
    --model_postfix qqp_Gen_4000
    
echo "/////////////////////////////////////////////////////////"

python eval.py \
    --test_data data/qqp_paragen_test.json \
    --num_beam_groups 12 \
    --model_store_path checkpoints \
    --model_postfix qqp_Gen_BCDist_4000