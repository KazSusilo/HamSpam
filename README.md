# HamSpam
![image](https://github.com/user-attachments/assets/0f752677-f3ff-4ff5-b3f6-321149729ee6)

## Summary
HamSpam is a deep learning model built to detect and filter spam messages with high accuracy using attention-based neural networks. Trained on the [UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), the model distinguishes between ham (legitimate) and spam (unwanted) messages using tokenized message content and custom attention layers. After fine-tuning key hyperparameters, the model achieves 99.2% test accuracy.

## Model
The architecture integrates both recurrent and attention mechanisms, designed to process and interpret sequential text data effectively:

* **Input Layer**: Accepts sequences of tokens
* **Embedding Layer**: Transforms each token into a dense 128-dimensional vector
* **LSTM Layer**: Captures temporal patterns and contextual information in message sequences
* **Attention Layer**: Enables the model to focus on the most relevant parts of the sequence output from the LSTM
* **Global Max Pooling**: Aggregates the most salient features across the time steps
* **Dense Layer**: Fully connected layer for high-level feature extraction
* **Output Layer**: Output a binary classification in one-hot encoding
  * ham ➔ `[1, 0]`
  * spam ➔ `[0, 1]`

The model is compiled using the Adam optimizer and trained using categorical crossentropy loss.

## Fine-tuning
To improve performance, four hyperparameters were systematically tested:

* Size of Attention Layers
* Number of Attention Layers
* Learning Rate (α)
* Number of Training Epochs

In each experiment, one hyperparameter was varied while others were held at baseline values. The final model combines the optimal values found in each individual test.

### Size of Attention Layers Experiment
![SoAL](https://github.com/user-attachments/assets/c158537f-6276-472d-9bde-4791eeb15c3c)

Baseline: `64`

Values Tested: `[16, 32, 64, 128, 256]`

Result: The model performed best with `32` attention units, striking a balance between complexity and generalization.

### Number of Attention Layers Experiment
![NoAL](https://github.com/user-attachments/assets/08c17cc4-2ec2-41c7-88a9-84d97740eeca)

Baseline: `1`

Values Tested: `[1, 2, 3, 4, 5]`

Result: No improvement was observed beyond `1` layer; additional layers led to overfitting and slower convergence.

### Values of α (Learning Rate) Experiment
![VoA](https://github.com/user-attachments/assets/078ae318-e377-4a43-9abc-cedad56d02bd)

Baseline: `.01`

Values Tested: `[.1, .01, .001, .0001, .00001]`

Result: A lower learning rate of `.0001` led to the most stable convergence and highest validation accuracy.

### Number of Epochs Experiment
![NoE](https://github.com/user-attachments/assets/ad210a79-a725-431d-a2af-bb39b06341e5)

Baseline: `10`

Values Tested: `[1, 5, 10, 25, 50]`

Result: `5` epochs yielded the best validation accuracy, with longer training showing signs of overfitting.

## Final Model Performance
| Dataset     | Loss                 | Accuracy          |
| ------------| -------------------- | ----------------- |
| Training    | 0.018000619485974312 | .9953846335411072 |
| Validation  | 0.055760711431503296 | .9856459498405457 |
| Test        | 0.027326082810759544 | .9916267991065979 |
