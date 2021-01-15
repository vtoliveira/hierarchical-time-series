import pandas as pd
import numpy as np 

class Reconciliation:

    def __init__(self, method, hierarchy):
        self.__check_method_is_valid(method)
        self.method = method
        self.hierarchy = hierarchy

    def __check_method_is_valid(self, method):
        valid_methods = ['BU', 'TD', 'OLS']
        if method not in valid_methods:
            raise ValueError(f"{method} is invalid, please choose one of {valid_methods}")

    def predict(self):
        if self.method == 'BU':
            



    