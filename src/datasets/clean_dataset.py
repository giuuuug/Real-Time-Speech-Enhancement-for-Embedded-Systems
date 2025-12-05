from enum import Enum
from datasets.voice_bank_demand_dataset import VoiceBankDemandDataset
import utils.directories as dir_helper

class CleanDatasetPathEnum(Enum):
    TRAIN_CLEAN_DATASET_PATH = "dataset/voicebank_demand/clean_trainset_28spk_wav"
    TEST_CLEAN_DATASET_PATH = "dataset/voicebank_demand/clean_testset_wav"


class CleanDataset(VoiceBankDemandDataset):
    def __init__(self, dataset:CleanDatasetPathEnum, transform):
        dir_helper.ensure_dir_not_empty(dataset)
        super().__init__(dataset, transform)
