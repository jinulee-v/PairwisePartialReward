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
suffix = args.suffix + '_seed' + str(args.seed)
bs = args.bs
gpu = args.gpu
lenpen = args.lenpen






##### fixed #####

model_name = '_'.join([base_model, dataset, suffix])
file = [file for file in files if model_name in file]
print(file)
# assert len(file) == 1
file = file[-1]

#################






trainer_state_path = os.path.join('../checkpoints', file, 'trainer_state.json')
with open(trainer_state_path, 'r') as f:
    trainer_state = json.load(f)
best_ckpt = trainer_state['best_model_checkpoint']
best_ckpt = os.path.join(best_ckpt, 'pytorch_model.bin')

# checkpoint = 'checkpoint-17000'
# best_ckpt = os.path.join("checkpoints", file, checkpoint, 'pytorch_model.bin')




##### fixed #####
common = f'{gpu} {base_model} {dataset} {suffix}'
if args.sample:
    c1 = f'./inference.sh {common} {best_ckpt} "--lenpen {lenpen} --num_beams {bs} --sampling"'
    if bs != 16:
        c2 = f'./eval_ibleu.sh {common} sampling_bs{bs}_default result_sampling_bs{bs}.json'
    else:
        c2 = f'./eval_ibleu.sh {common} sampling_default result_sampling.json'
else:
    c1 = f'./inference.sh {common} {best_ckpt} "--lenpen {lenpen} --num_beams {bs}"'
    if bs != 16:
        c2 = f'./eval_ibleu.sh {common} bs{bs}_default result_bs{bs}.json'
    else:
        c2 = f'./eval_ibleu.sh {common} default result.json'

print(c1)
print(c2)
os.system(c1)
os.system(c2)
