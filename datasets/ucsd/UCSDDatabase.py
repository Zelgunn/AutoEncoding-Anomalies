from datasets.ucsd.UCSDDataset import UCSDDataset
import os

from scheme import Database


class UCSDDatabase(Database):
    def load(self, database_path, **kwargs):
        if database_path.endswith("\\"):
            database_path = database_path[:-1]
        self.database_path = database_path

        self.train_dataset = self.load_dataset("Train")
        self.test_dataset = self.load_dataset("Test")

        if not self.train_dataset.saved_to_npz:
            self.normalize()
            self.train_dataset.save_to_npz()
            self.test_dataset.save_to_npz()

    def load_dataset(self, dataset_name: str):
        dataset_path = os.path.join(self.database_path, dataset_name)
        dataset = UCSDDataset(dataset_path=dataset_path)
        return dataset
