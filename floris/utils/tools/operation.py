import json
import numpy as np
import pandas as pd



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                  OPERATIONS                                  #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def json_load(path):
    with open(path,'r+') as load_f:
         config = json.load(load_f)
    return config


if __name__ == "__main__":
    pass