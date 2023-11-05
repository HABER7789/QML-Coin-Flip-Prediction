import tkinter as tk
from qiskit import QuantumCircuit, Aer, execute
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def apply_styles(widget, font_size=16, text_color="white", background_color="green", relief="raised", border_width=2, highlight_color="black"):
    widget.configure(
        font=("Helvetica", font_size),           # Set font and size
        foreground=text_color,                   # Text color
        background=background_color,             # Background color
        relief=relief,                           # Button relief style
        borderwidth=border_width,                # Border width
        highlightbackground=highlight_color      # Highlight color
    )

# Create a Quantum Circuit with a single Qubit
qc = QuantumCircuit(1, 1)
simulator = Aer.get_backend('qasm_simulator')

def generate_quantum_coin_flip():
    qc.h(0)
    qc.measure(0, 0)
    
    job = execute(qc, simulator, shots=1)
    result = job.result()
    counts = result.get_counts()
    measurement = list(counts.keys())[0]
    
    if measurement == '0':
        outcome = "Heads or measured bit is 0"
        
    else:
        outcome = "Tails or measured bit is 1"    
    
    result_label.config(text=f"Quantum Coin Flip Results: {outcome}")

def train_and_predict_model():
    dataset = []
    for _ in range(100):
        job = execute(qc, simulator, shots=1)
        result = job.result()
        counts = result.get_counts()
        measurement = list(counts.keys())[0]
        dataset.append(int(measurement))
    
    X = np.array(dataset).reshape(-1, 1)
    y = X
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    accuracy_text = f"Model Accuracy: {accuracy * 100:.2f}%"
    predictions_text = f"Predictions: {y_pred}"
    result_label.config(text=f"{accuracy_text}\n{predictions_text}")

    plt.hist(dataset, bins=[0, 1, 2], align='left', rwidth=0.9)
    plt.xticks([0, 1])
    plt.xlabel('Outcomes')
    plt.ylabel('Frequency')
    plt.title('Quantum Coin Flip Outcomes')
    plt.show()

root = tk.Tk()
root.title("Quantum Coin Flip User Interface")

generate_button = tk.Button(root, text="Flip a coin, the Quantum Way!", command=generate_quantum_coin_flip)
apply_styles(generate_button, background_color="#03045e")
generate_button.pack(pady=10)

train_button = tk.Button(root, text="Train and Predict Model", command=train_and_predict_model)
apply_styles(train_button, font_size=16, background_color="#0077b6")
train_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 16), background="#00b4d8", foreground="white")
result_label.pack(pady=10)

root.geometry("400x300")
root.mainloop()