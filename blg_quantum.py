# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import numpy as np

# data = load_iris()
# X = data.data
# y = data.target

# # Use only two classes for binary classification
# X = X[y != 2]
# y = y[y != 2]

# # Scale features 
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X) 

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)



# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# clf = SVC(kernel='rbf')
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print("Classical SVM Accuracy:", accuracy_score(y_test, predictions))



# import numpy as np
# from qiskit import BasicAer
# from qiskit.utils import QuantumInstance
# from qiskit.circuit.library import ZZFeatureMap
# from qiskit_machine_learning.algorithms import QSVC
# from qiskit_machine_learning.kernels import FidelityQuantumKernel
# from qiskit.primitives import Sampler
# from qiskit.algorithms.state_fidelities import ComputeUncompute

# # Sample data
# X_train = np.array([[0.1, 0.2], [0.4, 0.5]])
# y_train = np.array([0, 1])
# X_test = np.array([[0.3, 0.3], [0.6, 0.7]])

# # Feature map
# feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')

# # Kernel
# sampler = Sampler()
# fidelity = ComputeUncompute(sampler=sampler)
# quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# # QSVC model
# qsvc = QSVC(quantum_kernel=quantum_kernel)
# qsvc.fit(X_train, y_train)
# predictions = qsvc.predict(X_test)

# print("Predictions:", predictions)



