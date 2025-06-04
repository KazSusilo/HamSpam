import matplotlib.pyplot as plt

from . import experimental_model as emodel
from trainer import train_model


# Plotting function comparing training and validation performance
def plot_results(results, title, x_label, y_label):
  keys = list(results.keys())
  training_scores, validation_scores = [], []
  for values in results.values():
    # values = (training performance[loss, accuracy], validation performance[loss, accuracy])
    training_scores.append(values[0][1])
    validation_scores.append(values[1][1])

  plt.figure()
  plt.plot(keys, training_scores, label="Training Performance", marker='x')
  plt.plot(keys, validation_scores, label="Validation Performance", marker='o')
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.legend()
  plt.show()


# Experiment on the size of the attention layers
def attention_layers_size_experiment(size_of_attention_layers):
  experiment = "Experiment SoAL"
  x = "Size of Attention Layers"
  y = "Performance"
  experiment_helper(size_of_attention_layers, experiment, x, y)

# Experiement on the number of attention layers
def attention_layers_number_experiment(number_of_attention_layers):
  experiment = "Experiment NoAL"
  x = "Number of Attention Layers"
  y = "Performance"
  experiment_helper(number_of_attention_layers, experiment, x, y)

# Experiment on the value of the learning rate
def learning_rate_experiment(values_of_alpha):
  experiment = "Experiment VoA"
  x = "Values of Alpha"
  y = "Performance"
  experiment_helper(values_of_alpha, experiment, x, y)

# Experiement on the number of epochs of training
def epochs_experiment(number_of_epochs):
  experiment = "Experiment NoE"
  x = "Number of Epochs"
  y = "Performance"
  experiment_helper(number_of_epochs, experiment, x, y)

# Common code shared amongst all the experiments
def experiment_helper(options, experiment, x, y):
  results = {}
  for option in options:
    if ("SoAL" in experiment):
      model = emodel.experimental_ham_spam_model(soal=option)
      feedback = train_model(model)
    elif ("NoAL" in experiment):
      model = emodel.experimental_ham_spam_model(noal=option)
      feedback = train_model(model)
    elif ("VoA" in experiment):
      model = emodel.experimental_ham_spam_model(voa=option) 
      feedback = train_model(model)
    elif ("NoE" in experiment):
      model = emodel.experimental_ham_spam_model()
      feedback = train_model(model, epochs=option)
  
    model = feedback[0]
    training_performance = feedback[1]
    validation_performance = feedback[2]
    results[option] = (training_performance, validation_performance)
  plot_results(results, experiment, x, y)
    
# Best performing model using optimal parameters found in experiments
def optimal_model(size_of_attention_layers, number_of_attention_layers, epochs, alpha):
  model = emodel.experimental_ham_spam_model(soal=size_of_attention_layers, 
                                             noal=number_of_attention_layers, 
                                             voa=alpha)
  results = train_model(model, epochs=epochs)
  model = results[0]
  training_performance = results[1]
  validation_performance = results[2]
  test_performance = results[3]
  print("Training Performance: ", training_performance)
  print("Validation Performance: ", validation_performance)
  print("Test Performance: ", test_performance)


def experiments():
  # UNCOMMENT TO RUN SPECIFIC EXPERIEMENT
  # a. Experiment on the size of the attention layers
  sizes_of_attention_layers = [16, 32, 64, 128, 256]    # Experiment options
  #attention_layers_size_experiment(sizes_of_attention_layers)

  # b. Experiment on the number of attention layers
  numbers_of_attention_layers = [1, 2, 3, 4, 5]         # Experiment options
  #attention_layers_number_experiment(numbers_of_attention_layers)

  # c. Experiment on the value of the learning rate
  values_of_alpha = [.1, .01, .001, .0001, .00001]      # Experiment options
  #learning_rate_experiment(values_of_alpha)

  # d. Experiment on the number of epochs of training
  numbers_of_epochs = [1, 5, 10, 25, 50]                # Experiment options
  #epochs_experiment(numbers_of_epochs)

  # Optimal parameters based on experiments
  size_of_attention_layers = 32     # Base: 64
  number_of_attention_layers = 1    # Base: 1
  epochs = 5                        # Base: 10
  alpha = .0001                     # Base: .01
  optimal_model(size_of_attention_layers, number_of_attention_layers, epochs, alpha)


# Driver code
if __name__ == '__main__':
  experiments()