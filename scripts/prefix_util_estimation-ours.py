import json
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--suffix", type=str, required=True)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--lenpen", type=float, default=1.0)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()






files = os.listdir('../checkpoints')
base_model = args.base_model
dataset = args.dataset
datasetfile = f"data/{args.dataset}_paragen_test.json"
suffix = args.suffix + '_seed' + str(args.seed)
bs = args.bs
gpu = args.gpu
lenpen = args.lenpen






##### fixed #####

model_name = '_'.join([base_model, dataset, suffix])
print(model_name)
file = [file for file in files if model_name in file]
print(file)
# assert len(file) == 1
file = file[-1]

#################






trainer_state_path = os.path.join('../checkpoints', file, 'trainer_state.json')
with open(trainer_state_path, 'r') as f:
    trainer_state = json.load(f)
best_ckpt = trainer_state['best_model_checkpoint']
# best_ckpt = os.path.join("checkpoints", file, "checkpoint-27500")
best_ckpt = os.path.join(best_ckpt, 'pytorch_model.bin')

# checkpoint = 'checkpoint-17000'
# best_ckpt = os.path.join("checkpoints", file, checkpoint, 'pytorch_model.bin')




##### fixed #####
common = f'--base_model {base_model} --test_data {datasetfile} --model_postfix {suffix}'
c1 = f'CUDA_VISIBLE_DEVICES={gpu} python eval_partial_utility.py {common} --model_path {best_ckpt} --result_path results/{model_name}'

print(c1)
os.chdir("..")
os.system(c1)
