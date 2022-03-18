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
from .utils import Coms2SenseSingleSentenceExample
from transformers import (
    AutoTokenizer,
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Com2SenseDataProcessor(DataProcessor):
    """Processor for Com2Sense Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

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

        ##################################################
        # TODO: Use json python package to load the data
        # properly.
        # We recommend separately storing the two
        # complementary statements into two individual
        # `examples` using the provided class
        # `Coms2SenseSingleSentenceExample` in `utils.py`.
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # Make sure to add to the examples strictly
        # following the `_1` and `_2` order, that is,
        # `sent_1`'s info should go in first and then
        # followed by `sent_2`'s, otherwise your test
        # results will be messed up!
        # For the guid, simply use the row number (0-
        # indexed) for each data instance, i.e. the index
        # in a for loop. Use the same guid for statements
        # coming from the same complementary pair.
        # Make sure to handle if data do not have
        # labels field.
        json_path = os.path.join(data_dir, split+".json")
        data = json.load(open(json_path, "r"))

        examples = []

        for i in range(len(data)):
            datum = data[i]
            guid = i
            
            sentence1 = datum["sent_1"]
            sentence2 = datum["sent_2"]
            
            try:
                label1 = (datum["label_1"] == "True")
                label2 = (datum["label_2"] == "True")
            except:
                label1 = None
                label2 = None 

            domain = datum["domain"]
            scenario = datum["scenario"]
            numeracy = (datum["numeracy"] == "True")

            example1 = Coms2SenseSingleSentenceExample(
                guid=guid,
                text=sentence1,
                label=label1,
                domain=domain,
                scenario=scenario,
                numeracy=numeracy
            )

            example2 = Coms2SenseSingleSentenceExample(
                guid=guid,
                text=sentence2,
                label=label2,
                domain=domain,
                scenario=scenario,
                numeracy=numeracy
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


def domain_partition_analysis(examples, preds, domain):
    # returns the index of each example with the given domain
    indices = [i for i,x in enumerate(examples) if x.domain==domain]
    examples = [examples[i] for i in indices]
    preds = [preds[i] for i in indices]
    print("Number of {} examples: {}".format(domain, len(indices)))
    analysis(examples, preds)

def scenario_partition_analysis(examples, preds, scenario):
    # returns the index of each example with the given scenario
    indices = [i for i,x in enumerate(examples) if x.scenario==scenario]
    examples = [examples[i] for i in indices]
    preds = [preds[i] for i in indices]
    print("Number of {} examples: {}".format(scenario, len(indices)))
    analysis(examples, preds)

def analysis(val_examples, preds):
    guids = [x.guid for x in val_examples]
    labels = [x.label for x in val_examples]

    pair_acc = pairwise_accuracy(guids, preds, labels)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    print()
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(prec))
    print("Recall: {}".format(recall))
    print("F1 Score: {}".format(f1))
    print("Pairwise Accuracy: {}".format(pair_acc))
    print()

def pairwise_accuracy(guids, preds, labels):
    acc = 0.0  # The accuracy to return.
    
    guid_dict = {}
    for curr in range(len(guids)):
        curr_id = guids[curr]
        if curr_id not in guid_dict:
            guid_dict[curr_id] = True
        guid_dict[curr_id] = guid_dict[curr_id] and (preds[curr] == labels[curr])
    
    wrong = 0
    correct = 0
    for guid in guid_dict:
        if guid_dict[guid]:
            correct += 1
        else:
            wrong += 1

    acc = correct/(correct+wrong)

    return acc

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

if __name__ == "__main__":

    # set to None if don't intend to do analysis of prediction data. Otherwise, set to path to dev predictions txt file
    #analysis_data = "com2sense_only_real/com2sense_only_dev_predictions_third.txt"
    #analysis_data = "piqa_to_com2sense/piqa_to_com2sense_dev_predictions.txt"
    analysis_data = None

    # Test loading data.
    proc = Com2SenseDataProcessor(data_dir="datasets/com2sense")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()

    # Print Analysis
    if analysis_data is not None:
        
        print()
        print()

        file_path = os.path.join("dev_predictions", analysis_data)
        data = open(file_path, "r")
        contents = data.readlines()

        # convert strings of form '1\n' into the boolean True, and '0\n' into False
        preds = [x.strip()=='1' for x in contents]

        print("Overall Analysis:")
        print("Number of examples: {}".format(len(val_examples)))
        analysis(val_examples, preds)

        scenarios = unique([x.scenario for x in val_examples])
        for scenario in scenarios:
            print("Scenario: " + scenario)
            scenario_partition_analysis(val_examples, preds, scenario)

        domains = unique([x.domain for x in val_examples])
        for domain in domains:
            print("Domain: " + domain)
            domain_partition_analysis(val_examples, preds, domain)


        

    # print()
    # for i in range(3):
    #     print(val_examples[i])
    # print()
    # print(len(val_examples))
    # print(val_examples[0].label)
    # print(val_examples[0].guid)
    # print(val_examples[len(val_examples)-1].guid)
