import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--result_dir", type=str, default="results")
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()


