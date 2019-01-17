from abc import ABC, abstractmethod


class AutoEncoderScheme(ABC):
    def __init__(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def preprocess_data(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

# Create Database (Train + Test [+ Evaluation])
# # Do not load datasets

# Create Preprocessing Graph

# Preprocessing Queue

