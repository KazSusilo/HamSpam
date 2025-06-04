import tensorflow as tf

from config import MAX_LEN
from config import VOCAB_SIZE

def experimental_ham_spam_model(soal=64, noal=1, voa=0.01):
    """
    Supports experiments to fine-tune the hyperparameters for the 
    ham_spam_model

    Parameters:
        soal (int)  : size of attention layers
        noal (int)  : number of attention layers
        voa  (float): values of alpha (learning rate)

    Return:
        model: compiled model

    Network Architecture:
        input_layer: Accepts sequences of tokens of fixed length MAX_LEN
        embedding_layer: Maps each token to a dense vector of size 128
        lstm_layer: Processes the embedded sequence and ouputs a sequence (return_sequences=True)
        attention_layer: Applies self-attention over the LSTM output sequence
        pooled_layer: Applies global max pooling to reduce sequence to fixed-length vector
        dense_layer: Fully connected layer for high-level feature extraction
        output layer: Outputs probability distribution over 2 classes (ham, spam)
    """

    # Network architecture
    input_layer = tf.keras.Input(shape=(MAX_LEN,), name="input_layer")
    embedding_layer = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN, name="embedding_layer")(input_layer) 
    lstm_layer = tf.keras.layers.LSTM(soal, return_sequences=True, name="lstm_layer")(embedding_layer) 
    attention_input = lstm_layer
    for i in range (noal): # support various soal (powers of 2) and noal (minimum 1)
        query = tf.keras.layers.Dense(soal, name=f"query_{i}")(attention_input)
        key = tf.keras.layers.Dense(soal, name=f"key_{i}")(attention_input)
        value = tf.keras.layers.Dense(soal, name=f"value_{i}")(attention_input)
        attention_output = tf.keras.layers.Attention(name=f"attention_layer_{i}")([query, key, value])
        attention_input = attention_output  # feed into next attention layer if any
    pooling_layer = tf.keras.layers.GlobalMaxPooling1D(name="pooling_layer")(attention_output) 
    dense_layer = tf.keras.layers.Dense(16, activation="relu", name="hidden_layer")(pooling_layer) 
    output_layer = tf.keras.layers.Dense(2, activation="softmax", name="output_layer")(dense_layer)

    # model is a keras attention-based model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="Ham_Spam_Model")

    # Compile the model with ADAM optimizer and CCE loss
    optimizer_type = tf.keras.optimizers.Adam(learning_rate=voa)
    loss_type = "categorical_crossentropy"
    model.compile(optimizer=optimizer_type, loss=loss_type, metrics=["accuracy"])

    return model