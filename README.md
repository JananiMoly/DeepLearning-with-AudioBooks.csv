# DeepLearning-with-AudioBooks.csv
#import-libraries
import numpy as np
import pandas as pd 
import tensorflow as tf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#import-dataset
dataset = np.loadtxt(r'C:\Users\ADMIN\Desktop\Audiobooks-data.csv', delimiter=',')
#define-inputs and targets
inputs = dataset[:,0:11]
outputs = dataset[:,-1]

#balancing the priors.
np.unique(outputs, return_counts = 1)
num_one_targets = int(np.sum(outputs))
zero_counter = 0
index =[]

for i in range(outputs.shape[0]):
    if outputs[i] == 0:
        zero_counter = zero_counter+1
        if zero_counter > num_one_targets:
            index.append(i)
        
inputs_balanced = np.delete(inputs, index, axis =0)
outputs_balanced = np.delete(outputs, index, axis =0)

#featuring-scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
scaled_inputs = sc_x.fit_transform(inputs_balanced)

#shuffling the data          
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_outputs = outputs_balanced[shuffled_indices]

#splitingthe dataset into train, validation and test
samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count= samples_count-train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_outputs = shuffled_outputs[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_outputs = shuffled_outputs[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_outputs = shuffled_outputs[train_samples_count+validation_samples_count:]


#learning to save the npz file to access later
np.savez('AudioBooks training data', inputs =train_inputs, targets = train_outputs)
np.savez('AudioBooks validation data', inputs = validation_inputs, targets = validation_outputs)
np.savez('AudioBooks testing data', inputs =test_inputs, targets = test_outputs)

#importtensorfflow library for deep learning
import tensorflow as tf

#loading the saved npz file as train, validation and test inputs & targets
npz = np.load('Audiobooks_training_data.npz')

train_inputs = npz['inputs'].astype(float)
train_outputs=npz['targets'].astype(np.int)

npz = np.load('Audiobooks_validation_data.npz')

validation_inputs = npz['inputs'].astype(float)
validation_outputs=npz['targets'].astype(np.int)

npz = np.load('Audiobooks_test_data.npz')

test_inputs = npz['inputs'].astype(float)
test_outputs = npz['targets'].astype(np.int)

#building a deep learning network
input_size = 10 
output_size =2
hidden_layers_size = 30

#defining the model
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layers_size, activation ='relu'), 
                            tf.keras.layers.Dense(hidden_layers_size, activation='relu'), 
                            tf.keras.layers.Dense(output_size, activation = 'softmax')
                                          ])

#defining the optimization and loss function
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size =100
max_epochs = 100

#setting early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(patience =2)

#fitting the model to the dataset
model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=max_epochs, callbacks = [early_stopping], validation_data = (validation_inputs, validation_outputs), verbose =2)

#model accuracy
test_loss, test_accuracy = model.evaluate(test_inputs, test_outputs)
