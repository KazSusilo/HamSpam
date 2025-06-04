import tensorflow as tf
import numpy as np
import pandas as pd 
import math
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import MAX_LEN
from config import VOCAB_SIZE
from config import DATA_FILEPATH

# Rows from the dataset to use for: 
from config import training_rows    # training
from config import validation_rows  # validation
from config import test_rows        # testing


class MessageDataGenerator(tf.keras.utils.Sequence):
    """Keras data generator for the SMS Spam Collection Dataset"""
    def __init__(self, rows, batch_size):
        self.batch_size = batch_size
        self.rows = rows  # rows of data to use from the file

        # Read the data
        self.data = pd.read_csv(DATA_FILEPATH, sep="\t", header=None, names=["label", "message"])

        # One-hot encoding
        self.data["label"] = self.data["label"].map({"ham": [1,0], "spam":[0,1]})

        # Tokenize
        self.tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.data["message"]) # fit on selected rows
        self.indexes = np.arange(len(self.data))          # for shuffling

    def __len__(self):
        # Total number of batches used for one epoch
        batches_per_epoch = math.ceil(len(self.rows)/self.batch_size)
        return batches_per_epoch

    def __getitem__(self, index):
        # 'index' refers to the batch index to be retrieved
        batch_indexes = self.rows[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data.iloc[batch_indexes]

        # Tokenize, pad, and one-hot encode
        # x is one batch of data
        # y is the labels for the batch of data
        x = self.tokenizer.texts_to_sequences(batch_data["message"])
        x = pad_sequences(x, maxlen=MAX_LEN, padding='post', truncating='post')
        y = np.array(batch_data["label"].tolist())  # one-hot encoded labels

        return x, y


def train_model(model, epochs=10):
    training_gen = MessageDataGenerator(training_rows, 20)
    validation_gen = MessageDataGenerator(validation_rows, 20)
    test_gen = MessageDataGenerator(test_rows, 20)

    # Train model
    model.fit(training_gen, validation_data=validation_gen, epochs=epochs, verbose=1)

    # Performance of the model on the given data_gen in the form: [loss, accuracy]
    training_performance = model.evaluate(training_gen)
    validation_performance = model.evaluate(validation_gen)
    test_performance = model.evaluate(test_gen)

    return model, training_performance, validation_performance, test_performance