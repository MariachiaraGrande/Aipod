from abc import abstractmethod
from typing import List, Union

import numpy as np


class LeapHandler():
    def __init__(self):


        # Optimization params
        self.opt_vars = {}
        self.out_vars = {}
        self.opt_params = {}
        self.constraint_dict = {}




    @abstractmethod
    def init_parameters(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        raise NotImplementedError
    @abstractmethod
    def ask(self, **kwargs) -> dict:
        """

        :param kwargs:
        :return:
        """
        raise NotImplementedError

def convert_to_float(lst):
    result = []
    for item in lst:
        try:
            result.append(float(item))  # Attempt to convert to float
        except ValueError:
            result.append(item)  # If conversion fails, keep as string
    return result