# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 03:08:47 2023

@author: patel
"""
from qiskit import QuantumCircuit, Aer, execute

# first create a Quantum Circuit with a single Qubit
qc = QuantumCircuit(1,1) # 1 qubit and 1 classical bit

qc.h(0) #hadamard gate always converts a qubit into superposition

#measuring the qubit
qc.measure(0,0)

#Now we simulate the quantum circuit to obtain results
simulator = Aer.get_backend('qasm_simulator')
job = execute(qc, simulator, shots=1)
result = job.result()
counts = result.get_counts()
measurement = list(counts.keys())[0]

print(measurement)
"""
the list of result (current only one bit result will be 
stored in the list as shots=1

NOW WE GENERATE A DATASET where we store Multiple
Quantum Coin Flips in the dataset. 
"""
dataset = []
    
for i in range(100):
        job = execute(qc, simulator, shots=1)
        result = job.result()
        counts = result.get_counts()
        measurement = list(counts.keys())[0]
        dataset.append(int(measurement))
print(dataset)


"""
Alright, further we use a Machine Learning model 
and train it using the dataset we just created. With this,
we will able to predict the outcome of Quantum Coin Flip 
using Classical Machine Learning
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

X = np.array(dataset).reshape(-1, 1) #we have to reshape the data becasue Decision tree model requires data in 2D format 

y = X #same as input, because we are trying to predict the input itself (input output mapping)

#splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model selection and training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#now predict using the test dataset we split earlier
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100: .2f}%")

#visualization
plt.hist(dataset, bins=[0,1,2], align='left')
plt.xticks([0,1])
plt.xlabel('Outcomes')
plt.ylabel('Frequency')
plt.title('Quantum Coin Flip Outcomes')
plt.show()

#when you do the visulization the graph will represent the outcomes where the result was 0 or 1 with similar probabilities