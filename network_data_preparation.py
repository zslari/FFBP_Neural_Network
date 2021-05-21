from enum import Enum
from collections import deque
import numpy as np
import random


class DataMismatchError(Exception):
    pass


class NNData:
    """ Helps us manage our training and test data """

    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(factor):
        return min(1, max(factor, 0))

    def __init__(self, features=None, labels=None, train_factor=0.9):
        self._train_factor = NNData.percentage_limiter(train_factor)
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        try:
            self.load_data(features=features, labels=labels)
        except(ValueError, DataMismatchError):
            self._features = None
            self._labels = None
        self.split_set()

    def load_data(self, features=None, labels=None):
        """ Checks the length of features and labels then creates
        NumPy arrays from our lists of lists
        """
        if features is None or labels is None:
            features = []
            labels = []
        if len(features) != len(labels):
            raise DataMismatchError("Label and example lists have different "
                                    "lengths")
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = []
            self._labels = []
            raise ValueError("Label and example lists must be homogeneous and"
                             "numeric lists of lists")

    def split_set(self, new_train_factor=None):
        """ Splits the total number of examples into a testing set
        and a training set
        """
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        num_training_examples = int(len(self._features) * self._train_factor)
        num_test_examples = int(len(self._features) * (1-self._train_factor))
        base_list = [x for x in range(num_training_examples+num_test_examples)]
        self._train_indices = random.sample(base_list, k=num_training_examples)
        self._test_indices = [i for i in base_list if i not in self._train_indices]
        self._train_indices.sort()
        self._test_indices.sort()

    def prime_data(self, target_set=None, order=None):
        """ Loads one or more deques to be used as indirect indices """
        if target_set is None:
            self._train_pool = deque(self._train_indices[:])
            self._test_pool = deque(self._test_indices[:])
        if target_set == NNData.Set.TRAIN:
            self._train_pool = deque(self._train_indices[:])
        if target_set == NNData.Set.TEST:
            self._test_pool = deque(self._test_indices[:])
        if order == NNData.Order.RANDOM:
            self._train_pool = deque(random.sample(self._train_pool, len(self._train_pool)))
            self._test_pool = deque(random.sample(self._test_pool, len(self._test_pool)))

    def get_one_item(self, target_set=None):
        """ Returns one feature and its corresponding label """
        try:
            if target_set == NNData.Set.TEST:
                index = self._test_pool.popleft()
            else:
                index = self._train_pool.popleft()
            return self._features[index], self._labels[index]
        except IndexError:
            return None

    def number_of_samples(self, target_set=None):
        """ Returns total number of testing examples """
        if target_set is NNData.Set.TEST:
            return len(self._test_indices)
        elif target_set is NNData.Set.TRAIN:
            return len(self._train_indices)
        else:
            return len(self._features)

    def pool_is_empty(self, target_set=None):
        """ Checks if the targeted pool is empty """
        if target_set is NNData.Set.TEST:
            return len(self._test_pool) == 0
        else:
            return len(self._train_pool) == 0
