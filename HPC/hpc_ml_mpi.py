from mpi4py import MPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load dataset on all nodes
data = load_iris()
X, y = data.data, data.target

# Split data among processes
chunk_size = len(X) // size
start = rank * chunk_size
end = (rank + 1) * chunk_size if rank != size - 1 else len(X)

X_chunk = X[start:end]
y_chunk = y[start:end]

# Train local Decision Tree
model = DecisionTreeClassifier()
model.fit(X_chunk, y_chunk)

# Predict on full data
predictions = model.predict(X)

# Gather predictions at root
all_predictions = comm.gather(predictions, root=0)

if rank == 0:
    # Combine predictions (majority voting)
    all_predictions = np.array(all_predictions)
    final_predictions = []
    for i in range(all_predictions.shape[1]):
        votes = np.bincount(all_predictions[:, i])
        final_predictions.append(np.argmax(votes))

    accuracy = accuracy_score(y, final_predictions)
    print(f"Distributed Decision Tree Accuracy: {accuracy:.2f}")


# pip install mpi4py scikit-learn
# mpiexec -n 4 python hpc_ml_mpi.py
