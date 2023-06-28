base_model=$2 # e.g., t5, bart
dataset=$3 # e.g., qqp, mscoco
postfix=$4
checkpoint=$5
args="$6"
model=${base_model}_${dataset}_${postfix}
output_dir=results/${model}

cd ..

if [ ! -d $output_dir ];then
    mkdir -p $output_dir
fi

CUDA_VISIBLE_DEVICES=$1 python inference.py \
  --test_data data/${dataset}_paragen_test.json \
  --batch_size 24 \
  --model_path ${checkpoint} \
  --result_path $output_dir \
  --model_postfix $model \
  --base_model $base_model \
  ${args}