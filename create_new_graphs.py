import argparse
from graphbench.datasets.sat import SATDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="/storage/work/graph_bench/dataset_test")
parser.add_argument("--name", type=str, default="sat_lcg_as")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--generate", type=bool, default=True)
args = parser.parse_args()

SATDataset(name=args.name, split=args.split, root=args.root, generate=args.generate)