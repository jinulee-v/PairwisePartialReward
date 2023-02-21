"""
Author: Jinu Lee
Credits:
- Dependency parsing:  https://github.com/Unipisa/diaparser
- Tree Kernel scoring: https://github.com/mawilliams7/pyftk
"""

from argparse import ArgumentParser
import json
from tqdm import tqdm, trange
import os, sys
import logging

import torch
import math

import nltk
from nltk.parse.dependencygraph import DependencyGraph
from diaparser.parsers import Parser
from tokenizer.tokenizer import Tokenizer # from diaparser

logger = None
dependency_parser = Parser.load('en_ptb.electra', lang='en')
dependency_tokenizer = Tokenizer('en', verbose=False)

all_parse_trees = []

def diaparser_to_nltk_tree(sentences):
    # sentences: List[List[str]] : batch of tokenized sentences
    result = dependency_parser.predict(sentences, text='en') # Parse sentences
    result = [str(sent) for sent in result.sentences] # Get CoNLL-U (10 cols) representation
    result = ['\n'.join([line for line in conll.split('\n') if not line.startswith('#')]) for conll in result] # remove comment lines since NLTK cannot interpret it

    dep_tree = [
        DependencyGraph(conll, top_relation_label='root').tree() # Convert to nltk.tree.Tree
        for conll in result
    ]

    return dep_tree

class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]

def extract_production_rules(tree, production_rules):
    left_side = tree.label()
    right_side = ""
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            right_side = right_side + " " + subtree.label()
            extract_production_rules(subtree, production_rules)
        else:
            right_side = right_side + " " + subtree
    production_rules.append((left_side + " ->" + right_side, tree))

def find_node_pairs(first_tree, second_tree):
    node_pairs = set()
    first_tree_production_rules = list()
    extract_production_rules(first_tree, first_tree_production_rules)
    first_tree_production_rules = sorted(first_tree_production_rules, key=lambda x : x[0])
    second_tree_production_rules = list()
    extract_production_rules(second_tree, second_tree_production_rules)
    second_tree_production_rules = sorted(second_tree_production_rules, key=lambda x : x[0])
    node_1 = first_tree_production_rules.pop(0)
    node_2 = second_tree_production_rules.pop(0)
    while node_1[0] != None and node_2[0] != None:
        if node_1[0] > node_2[0]:
            if len(second_tree_production_rules) > 0:
                node_2 = second_tree_production_rules.pop(0)
            else:
                node_2 = [None]
        elif node_1[0] < node_2[0]:
            if len(first_tree_production_rules) > 0:
                node_1 = first_tree_production_rules.pop(0)
            else:
                node_1 = [None]
        else:
            while node_1[0] == node_2[0]:
                second_tree_production_rules_index = 1
                while node_1[0] == node_2[0]:
                    node_pairs.add((str(node_1[1]), str(node_2[1])))
                    if second_tree_production_rules_index < len(second_tree_production_rules):
                        node_2 = second_tree_production_rules[second_tree_production_rules_index]
                        second_tree_production_rules_index += 1
                    else:
                        node_2 = [None]
                if len(first_tree_production_rules) > 0:
                    node_1 = first_tree_production_rules.pop(0)
                else:
                    node_1 = [None]
                if len(second_tree_production_rules) > 0:
                    node_2 = second_tree_production_rules[0]
                else:
                    node_2 = [None]
                if node_1[0] == None and node_2[0] == None:
                    break
    return node_pairs

@Memoize
def fast_tree_kernel(first_tree_index, second_tree_index):
    global all_parse_trees
    kernel_score = 0
    first_tree = all_parse_trees[first_tree_index]
    second_tree = all_parse_trees[second_tree_index]
    node_pairs = find_node_pairs(first_tree, second_tree)
    for node in node_pairs:
        if node[0] == node[1]:
            kernel_score += 1
    return kernel_score

def normalized_fast_tree_kernel(first_tree_index, second_tree_index):
    return fast_tree_kernel(first_tree_index, second_tree_index) / math.sqrt(fast_tree_kernel(first_tree_index, first_tree_index) * fast_tree_kernel(second_tree_index, second_tree_index))


def main(args):
    # Init logger
    assert os.path.isdir(args.model_store_path)
    log_path = os.path.join(args.model_store_path, args.model_postfix)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    if not args.secure:
        # Remove original log file
        if os.path.exists(os.path.join(log_path, "eval_treekernel.log")):
            os.remove(os.path.join(log_path, "eval_treekernel.log"))
    file_handler = logging.FileHandler(os.path.join(log_path, "eval_treekernel.log"))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Log basic info
    logger.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info("- %s: %r", arg, value)
    logger.info("")
    
    # Load generated data
    result_store_path = os.path.join(args.model_store_path, args.model_postfix, "result.json")
    with open(result_store_path, "r", encoding="UTF-8") as file:
        result = json.load(file)

    # Generate parse trees
    logger.info("Generate parse trees... (Electra-DiaParser trained for PTB)")
    all_sentences = [] # Temporary storage for sentences
    reference = [] # stores indices of all_parse_trees
    outputs = []
    for r in result:
        # Append tree for reference
        reference.append(len(all_sentences))
        all_sentences.append(r["input"])
        # Append trees for outputs
        output = []
        for sent in r["paraphrases"]:
            output.append(len(all_sentences))
            all_sentences.append(sent)
        outputs.append(output)

    # Batchified parsing
    for start in trange(0, len(all_sentences), args.batch_size):
        end = min(start + args.batch_size, len(all_sentences))
        tokenized_sents = [
            [token.text for token in dependency_tokenizer.predict(sent)[0].tokens]
        for sent in all_sentences[start:end]] # Tokenize and extract only lexical forms
        dep_trees = diaparser_to_nltk_tree(tokenized_sents)
        all_parse_trees.extend(dep_trees)
    assert len(all_parse_trees) == len(all_sentences)
    
    # Calculate tree kernel scores
    logger.info("Calculate Tree Kernel scores...")
    kernel_scores_per_beam = []
    for ref, outs in zip(tqdm(reference), outputs):
        scores = []
        for out in outs:
            scores.append(
                normalized_fast_tree_kernel(ref, out)
            )
        kernel_scores_per_beam.append(scores)
    kernel_scores_per_beam = torch.tensor(kernel_scores_per_beam)
    # num_beams * total_length

    logger.info("=================================================")
    logger.info("Analysis result")

    logger.info("")
    logger.info("Tree Kernel score(lower score -> dissimilar paraphrase)")
    logger.info(f"Total average: {torch.mean(kernel_scores_per_beam)}")
    logger.info(f"kernel_scores_per_beam score per beam:")
    for beam_id, score in enumerate(torch.mean(kernel_scores_per_beam, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")
    kernel_scores_per_beam_sorted, _ = torch.sort(kernel_scores_per_beam, dim=1)
    logger.info(f"kernel_scores_per_beam score per beam(sorted):")
    for beam_id, score in enumerate(torch.mean(kernel_scores_per_beam_sorted, dim=0).tolist()):
        logger.info(f"    beam {beam_id + 1}: {score}")

    logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=False, default='checkpoints', help="Directory to store model checkpoints.")
    parser.add_argument("--model_postfix", required=True)
    
    parser.add_argument("--batch_size", required=False, type=int, default=16, help="Batch size for dependency parsing")
    parser.add_argument("--gpu", required=False, type=int, default=0, help="GPU Id for dependency parsing. Only considered when `torch.cuda.is_available()` is True.")

    parser.add_argument("--secure", required=False, action="store_true", help="")

    args = parser.parse_args()
    main(args)
