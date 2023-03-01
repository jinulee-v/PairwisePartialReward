import argparse
import json
import random

SELECT = 100

def main(args):
    
    print("Command line args: checkpont names(model_postfix)")
    checkpoints = args.checkpoints
    print("- " + "\n- ".join(checkpoints))

    checkpoints_to_id = dict([(chkpt, i) for i, chkpt in enumerate(checkpoints)])
    paraphrases = {}

    # Gather data from all checkpoints
    for chkpt in checkpoints:
        with open(f"checkpoints/{chkpt}/result.json", "r", encoding="UTF-8") as file:
            data = json.load(file)
        for datum in data:
            # Gather sent
            input_sent = datum["input"]
            para_sent = datum["paraphrases"][0].strip()

            if input_sent not in paraphrases:
                paraphrases[input_sent] = ["" for _ in checkpoints]            
            paraphrases[input_sent][checkpoints_to_id[chkpt]] = para_sent
    
    # Select random data with all distinct paraphrases
    random.seed(0)
    paraphrases = list(paraphrases.items())
    random.shuffle(paraphrases)

    result = []
    for input_sent, paras in paraphrases:
        if len(set(paras)) == len(paras): # No duplicate paraphrases between chkpts
            result.append({
                "input": input_sent,
                "paraphrases": paras
            })
            if len(result) >= SELECT:
                break
    
    # Output the result data
    with open(args.output_file, "w", encoding="UTF-8") as file:
        json.dump({
            "checkpoints": checkpoints,
            "data": result
        }, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", "-c", action="append")
    parser.add_argument("--output_file")

    args = parser.parse_args()

    main(args)