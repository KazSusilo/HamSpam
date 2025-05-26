import os
from dotenv import load_dotenv
import pandas as pd 

MAX_LEN = 50
VOCAB_SIZE = 10000

# Load specific DATA_FILEPATH from .env file
load_dotenv()
DATA_FILEPATH = os.getenv("DATA_FILEPATH")

# Define training, validation, and test split
training_split = .7     # 70% Training
validation_split = .15  # 15% Validation
test_split = .15        # 15% Test

# Split dataset
dataset = pd.read_csv(DATA_FILEPATH, sep="\t", header=None, names=["label", "message"])
total_rows = len(dataset)
training_rows = range(0, int(training_split * total_rows))
validation_rows = range(int(training_split * total_rows), int((training_split + validation_split) * total_rows))
test_rows = range(int((training_split + validation_split) * total_rows), total_rows)