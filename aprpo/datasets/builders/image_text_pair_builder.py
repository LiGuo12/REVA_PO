import logging
from aprpo.common.registry import registry
from aprpo.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from aprpo.datasets.datasets.dataset import MIMIC_Dataset, IUXray_Dataset

@registry.register_builder("mimic")
class MIMIC_Builder(BaseDatasetBuilder):
    train_dataset_cls = MIMIC_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic/mimic.yaml",
    }

    def build_datasets(self):
        """Build train, validation, and test datasets"""
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        ann_file = build_info.get("ann_file", None)

        datasets = dict()
        # Create datasets for all splits
        splits = ['train', 'val', 'test']
        for split in splits:
            datasets[split] = self.train_dataset_cls(
                storage_path=storage_path,
                ann_file=ann_file,
                split=split
            )
        return datasets

@registry.register_builder("iuxray")
class IUXray_Builder(BaseDatasetBuilder):
    train_dataset_cls = IUXray_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/iuxray/iuxray.yaml",
    }

    def build_datasets(self):
        """Build train, validation, and test datasets"""
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        ann_file = build_info.get("ann_file", None)

        datasets = dict()

        # Create datasets for all splits
        splits = ['train', 'val', 'test']
        for split in splits:
            datasets[split] = self.train_dataset_cls(
                storage_path=storage_path,
                ann_file=ann_file,
                split=split,
            )
        return datasets
    