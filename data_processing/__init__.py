from .dummy_data import DummyDataProcessor
from .com2sense_data import Com2SenseDataProcessor
from .semeval_data import SemEvalDataProcessor
from .physicalIQA_data import PhysicalIQADataProcessor

from .processors import DummyDataset
from .processors import Com2SenseDataset
from .processors import SemEvalDataset
from .processors import PhysicalIQADataset

data_processors = {
    "dummy": DummyDataProcessor,
    "com2sense": Com2SenseDataProcessor,
    "semeval": SemEvalDataProcessor,
    "physicalIQA": PhysicalIQADataProcessor,
}


data_classes = {
    "dummy": DummyDataset,
    "com2sense": Com2SenseDataset,
    "semeval": SemEvalDataset,
    "physicalIQA": PhysicalIQADataset,
}
