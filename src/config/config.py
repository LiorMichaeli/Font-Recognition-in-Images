"""
This file contains the configuration for the project.

Author:
    Lior Michaeli
"""


# Import libraries
import torch


# Define memory device
MEMORY_DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
MEMORY_DEVICE = torch.device(MEMORY_DEVICE_NAME)


# Constants
NUM_OF_CLASSES_IN_FONT_DATASET = 7
BATCH_SIZE = 128
INPUT_SHAPE = (3, 32, 16)
NUMBER_OF_ORIG_IMGS_IN_TRAINING_SET_OF_CHARS = 745
TOTAL_NUMBER_OF_ORIG_IMGS = 829
TOP_LEFT = 0
TOP_RIGHT = 1
BOTTOM_RIGHT = 2
BOTTOM_LEFT = 3


# ID to font or font to ID mapping
map_dict_id_to_font = {
            0: 'Flower Rose Brush',
            1: 'Skylark',
            2: 'Sweet Puppy',
            3: 'Ubuntu Mono',
            4: 'VertigoFLF',
            5: 'Wanted M54',
            6: 'always forever'}

map_dict_font_to_id = {
            'Flower Rose Brush': 0,
            'Skylark': 1,
            'Sweet Puppy': 2,
            'Ubuntu Mono': 3,
            'VertigoFLF': 4,
            'Wanted M54': 5,
            'always forever': 6}