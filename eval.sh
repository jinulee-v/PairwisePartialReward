export TOKENIZERS_PARALLELISM=false

# python inference.py \
#     --test_data data/qqp_paragen_test.json \
#     --model_postfix bart_qqp_1gen
# python inference.py \
#     --test_data data/qqp_paragen_test.json \
#     --model_postfix bart_qqp_1gen_2gen+triecl
# python inference.py \
#     --test_data data/qqp_paragen_test.json \
#     --model_postfix bart_qqp_1gen+triecl

python eval.py --model_postfix bart_qqp_1gen
python eval.py --model_postfix bart_qqp_1gen_2gen+triecl
python eval.py --model_postfix bart_qqp_1gen+triecl