from enum import Enum
from src.datasets.voice_bank_demand_dataset import VoiceBankDemandDataset
from src.utils import directories as dir_helper

TRAIN_NOISY_DATASET_PATH = "dataset/voicebank_demand/noisy_trainset_28spk_wav"
TEST_NOISY_DATASET_PATH = "dataset/voicebank_demand/noisy_testset_wav"


class NoisyDataset(VoiceBankDemandDataset):
    def __init__(self, dataset: str, transform):
        dir_helper.ensure_dir_not_empty(dataset)
        super().__init__(dataset, transform)
