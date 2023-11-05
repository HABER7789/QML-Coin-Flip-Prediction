# QML-Coin-Flip-Prediction
Creation of a simple Quantum circuit that simulates a quantum coin flip and use classical ML model that predicts the outcome of a quantum coin flip. 

- Qiskit was to create a quantum circuit for a single qubit representing the quantum coin. The circuit has a single classical bit and a single qubit. 
- The qubits were then passed throught Hadamard gate which will help us to create a superposition situation for Heads (0) and Tails (1). The simulation of the quantum circuit is done prior to its measurement.
- Further, a dataset is created which has all the measured values of quantum coin flips stored in it. Several errors were encountered when trying to feed the data to a classical machine learning model.
- Then, select a classical ML model, DecisionTree which predicts the outcomes for the quantum coin flips. The dataset is first split into train and test set. In this scenario, the predicted output is hoped to be same as the input data (measurement of heads or tails). So, it is expected that the model accuracy will be high and almost a 100 percent precise.
- The project also has a simple Graphical User Interface (GUI) built inside the python code itself using the library 'tkinter' and finally a graphical representaion of the Outcomes.

The project was simple exploration of quantum computing and machine learning. The project is also relatively simpler in comparison to the complex quantum ML algorithms, however these are some baby steps that are to be done before jumping straight into complex algorithms. Feel free to use the code for learning purposes. 

I have attached three files, there are two python files, Quantum Coin Flip Prediction.py and QCFP GUI.py 
- Quantum Coin Flip Prediction.py: This file is a step by step approach to the project, with necessary comment lines for better understanding.
- QCFP GUI.py: This file has the code for GUI as well as the visualization of the outcomes, this does not have step by step comment lines, however it can be easily understood after going throught the first python file provided in the repository.
