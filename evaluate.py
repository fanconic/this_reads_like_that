import torch
from torch import nn
import yaml
import wandb
import argparse
import pandas as pd


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))

    main(config)
