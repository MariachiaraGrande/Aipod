import os
import sys
from typing import List

import yaml
import json
import pickle
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def rename_columns(filename: str, parameters: list, columns: list):
    ''' Rename columns in the given dataframe based on the provided parameters and
        columns list.'''
    data = pd.read_excel(filename)
    col = columns
    par = parameters
    for i, name in enumerate(par):
        data = data.rename(columns= {data.columns[i]:name})
    for j, name in enumerate(col):
        data = data.rename(columns={data.columns[len(par):][j]:name})
    return data
def evaluation_mean_std(filename: str):
     ''' Evaluate the mean and standard deviation of the design variables.'''
     filename= 'docs/data/doe_df_experiments.xlsx'
     data = pd.read_excel(filename)
     Ouput1_mean = data.iloc[:,lambda data:[4,6,8,10,12]].mean(axis=1)
     Output2_mean = data.iloc[:, lambda data:[5,7,9,11,13]].mean(axis=1)
     Ouput1_std = data.iloc[:, lambda data:[4,6,8,10,12]].std(axis=1)
     Output2_std = data.iloc[:, lambda data:[5,7,9,11,13]].std(axis=1)
     data['Output1'] = Ouput1_mean
     data['Ouput1_std'] = Ouput1_std
     data['Output2'] = Output2_mean
     data['Output2_std'] = Output2_std
     return data

def label_encoder(data: pd.DataFrame, target: list, encoded_target:list):
    ''' Encode the target variable astype int.'''
    for n,value in enumerate(target):
        data[encoded_target[n]] = (data[target[n]]!=0).astype('int')
    return data

def flatlist(t: list) -> List:
    """
    Flatten list

    :param t: list to be flatted
    :return: flatted list

    """
    return [item for sublist in t for item in sublist]


def get_query_folder() -> Path:
    cf = Path(__file__).parent.absolute()
    qdir = cf / 'queries'
    return qdir


def get_query_from_sql_file(
        file_name: str,
        kwargs: dict = None
) -> str:
    with open(get_query_folder() / file_name, 'r') as f:
        query = f.read()
    # injecting kwargs if not None
    if kwargs is not None:
        query = query.format(**kwargs)
    return query

def touch_dir(dir_path: str) -> None:
    """
    Create dir if not exist

    :param dir_path: directory path

    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


# CONSTANT
LOG_FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
OUTPUT_LOG_DIR = 'logs'
touch_dir(OUTPUT_LOG_DIR)
LOG_FILE = os.path.join(OUTPUT_LOG_DIR, f'{datetime.now().strftime("%Y%m%d")}.log')


# FILE HANDLER UTILS
def dump_excel(data: pd.DataFrame, path: str, filename: str) -> None:
    """
    Dump dataframe to excel file

    :param data: input dataframe
    :param path: output path
    :param filename: filename

    """
    data.to_excel(os.path.join(path, filename), index=False)


def copy_to_folder(file: str, directory: str) -> str:
    """
    Copy passed file to a new folder with the same name. The function automatically
    creates the output folder if it does not exist

    :param file: path of file to be copied
    :param directory: output directory
    :return: new path of the copied file

    """
    touch_dir(directory)
    r = Path(file)
    out_f = os.path.join(directory, r.name)
    try:
        shutil.copyfile(file, out_f)
    except shutil.SameFileError:
        pass
    return out_f


def move_to_folder(file: str, directory: str) -> str:
    """
    Move passed file to a new folder with the same name. The function automatically
    creates the output folder if it does not exist

    :param file: path of file to be moved
    :param directory: output directory
    :return: new path of the moved file

    """
    out_f = copy_to_folder(file, directory)

    try:
        os.remove(file)
    except FileNotFoundError:
        pass
    return out_f


# CONFIG AND LOGGING

def load_config(path_file: str) -> dict:
    """
    Load config file from yaml file

    :param path_file: folder containing the file
    :return: dict of the config file

    """
    with open(path_file, 'r', encoding='utf-8') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config


def load_pickle(path):
    """
    Load data from a pickle file
    :param path: current file directory and name
    :return: pickle object

    """
    obj = pickle.load(open(path, 'rb'))
    return obj


def write_config(data: dict, path_file: str) -> None:
    """
    Load config file from yaml file

    :param path_file: folder containing the file
    :param data: file name and extension
    :return: dict of the config file

    """
    with open(path_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, indent=4)


def load_json(path: str, name: str = None) -> dict:
    """
    Load config file from yaml file

    :param path: folder containing the file
    :param name: file name and extension
    :return: dict of the config file

    """
    if name:
        to_read = os.path.join(path, f'{name}')
    else:
        to_read = path
    with open(to_read, 'r') as fp:
        config = json.load(fp)
    return config


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dump_json(data: dict = {}, filepath: str = None) -> None:
    """
    Dump dict to json

    :param data: dict to be saved
    :param filepath: path of the storage file

    """
    with open(filepath, "w") as outfile:
        json.dump(data, outfile, cls=NpEncoder, indent=4)


def setup_logger(
        name: str,
        level: str = 'DEBUG',
) -> logging.Logger:
    """
    Setup logger

    :param name: logger name, generally the script or class name
    :param level: level of debugger
    :return logger: the logger with debug level in development (change into info in production)

    """
    logger = logging.getLogger(name.upper())

    logger.setLevel(level)
    logger.handlers = []
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger


def get_file_handler():
    """

    Handle log file


    """
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(LOG_FORMATTER)
    return file_handler


def get_console_handler():
    """
    Handle console

    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(LOG_FORMATTER)
    return console_handler
