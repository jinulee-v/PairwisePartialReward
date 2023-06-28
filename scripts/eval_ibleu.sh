# base_model=$2
# dataset=$3
cd ..

CUDA_VISIBLE_DEVICES=$1 python eval_bert_ibleu.py \
  --model_store_path results \
  --model_postfix ${2}_${3}_${4} \
  --eval_postfix $5 \
  --eval_file $6
