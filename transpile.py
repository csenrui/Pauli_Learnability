from turtle import back
from qiskit import IBMQ, QuantumCircuit
from qiskit.compiler import transpile
import sys
import numpy as np
from scipy.linalg import sqrtm

# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-research', group='berkeley-6', project='main')
# backend = provider.get_backend('ibmq_jakarta')

# gate_length = backend.properties().gate_length('sx',qubits=2)
# sys.exit(0)



qc = QuantumCircuit(2)
qc.barrier()
qc.id(0)
qc.id(1)
qc.barrier()
qc.id(0)
qc.x(1)
qc.barrier()
qc.id(0)
qc.y(1)
qc.barrier()
qc.id(0)
qc.z(1)
qc.barrier()
qc.id(0)
qc.x(1)
qc.sx(1)
qc.barrier()
qc.id(0)
qc.y(1)
qc.sx(1)
qc.barrier()
qc.id(0)
qc.z(1)
qc.sx(1)
qc.barrier()
qc.id(0)
qc.x(1)
qc.s(1)
qc.barrier()
qc.id(0)
qc.y(1)
qc.s(1)
qc.barrier()
qc.id(0)
qc.z(1)
qc.s(1)
qc.barrier()

# qc_transpile = transpile(qc,backend=backend,initial_layout=[1,2],optimization_level=3)
# backend.run(qc_transpile,shots=1000)
qc_transpile = transpile(qc,basis_gates=['cx', 'id', 'rz', 'sx', 'x'],optimization_level=3)
print(qc)
print(qc_transpile)

# X = np.array([[0,1],[1,0]])
# Y = np.array([[0,-1j],[1j,0]])
# Z = np.array([[1,0],[0,-1]])
# SX = sqrtm(X)
# print(SX.dot(X))
# print(-1j * Z.dot(SX).dot(Z))