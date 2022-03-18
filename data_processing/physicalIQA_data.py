import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
from tqdm import tqdm
from .utils import DataProcessor
from .utils import PhysicalIQASingleSentenceExample
from transformers import (
    AutoTokenizer,
)


class PhysicalIQADataProcessor(DataProcessor):
    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

        # TODO: Label to Int mapping, dict type.
        self.label2int = {"True": 1, "False": 0}

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        json_path = os.path.join(data_dir, split+".jsonl")
        #data = json.load(open(json_path, "r"))
        data_lines = []
        with open(json_path, 'r') as json_file:
            data_lines=json_file.read().splitlines()

        try:
            label_path = os.path.join(data_dir, split+"-labels.lst")
            label_data = []
            with open(label_path, 'r') as label_file:
                label_data=label_file.read().splitlines()
        except:
            pass

        examples = []

        for i in range(len(data_lines)):
            datum = json.loads(data_lines[i])
            guid = i
            
            goal = datum["goal"]
            sol1 = datum["sol1"]
            sol2 = datum["sol2"]
            
            try:
                label1 = (int)(label_data[i] == '0')
                label2 = (int)(label_data[i] == '1')
            except:
                label1 = None
                label2 = None

            example1 = PhysicalIQASingleSentenceExample(
                guid=guid,
                text=goal + ' ' + sol1,
                label=label1,
                goal=goal
            )

            example2 = PhysicalIQASingleSentenceExample(
                guid=guid,
                text=goal + ' ' + sol2,
                label=label2,
                goal=goal
            )

            examples.append(example1)
            examples.append(example2)


        # End of TODO.
        ##################################################

        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


if __name__ == "__main__":

    # Test loading data.
    proc = PhysicalIQADataProcessor(data_dir="datasets/physicalIQA")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    print("TRAIN EXAMPLES:")
    for i in range(3):
        print(train_examples[i])
    print("DEV EXAMPLES:")
    for i in range(3):
        print(val_examples[i])
    print("TEST EXAMPLES:")
    for i in range(3):
        print(test_examples[i])
    print()
