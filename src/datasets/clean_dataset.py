from enum import Enum
from src.datasets.voice_bank_demand_dataset import VoiceBankDemandDataset
from src.utils import directories as dir_helper

TRAIN_CLEAN_DATASET_PATH = "dataset/voicebank_demand/clean_trainset_28spk_wav"
TEST_CLEAN_DATASET_PATH = "dataset/voicebank_demand/clean_testset_wav"


class CleanDataset(VoiceBankDemandDataset):
    def __init__(self, dataset:str, transform):
        dir_helper.ensure_dir_not_empty(dataset)
        super().__init__(dataset, transform)
