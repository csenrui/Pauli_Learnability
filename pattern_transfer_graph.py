import numpy as np
from numpy.linalg import matrix_rank
import networkx as nx
import itertools, copy, random, sys, json
import qiskit
from qiskit import quantum_info
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliList

n = 2
CNOT = qiskit.circuit.library.CXGate()
SWAP = qiskit.circuit.library.SwapGate()
C = quantum_info.Clifford.from_circuit(SWAP)

qc = QuantumCircuit(2)
qc.cx(0,1)
qc.swap(0,1)
C = quantum_info.Clifford.from_circuit(qc)

C = quantum_info.random_clifford(n)

def pauli_to_int(p):
    n = len(p)
    s = ''
    for i in range(n):
        if p[i] == "I":
            s += '00'
        if p[i] == "X":
            s += '01'
        if p[i] == "Y":
            s += '10'
        if p[i] == "Z":
            s += '11'
    return int(s,2)

def int_to_pauli(n,t):
    s = format(t,'0%db' %(2*n))
    p = ''
    for i in range(n):
        if s[2*i:2*i+2] == '00':
            p += "I"
        if s[2*i:2*i+2] == '01':
            p += "X"
        if s[2*i:2*i+2] == '10':
            p += "Y"
        if s[2*i:2*i+2] == '11':
            p += "Z"
    return p

def pattern(pauli):
    pattern = ''
    for i in range(len(pauli)):
        if pauli[i] == 'I':
            pattern += '0'
        else:
            pattern += '1'
    return pattern

def commute(p,q):
    c = 1
    n = len(p)
    for i in range(n):
        if p[i] != 'I' and q[i] != 'I':
            if p[i] != q[i]:
                c *= -1
    return c

def hadamard(pauli):
    n = len(pauli)
    v = [0 for i in range(4**n)]
    paulis = list(itertools.product(['I','X','Y','Z'], repeat=n))
    paulis = [''.join(p) for p in paulis]
    for q in paulis:
        v[pauli_to_int(q)] = commute(pauli,q)
    return v


paulis = list(itertools.product(['I','X','Y','Z'], repeat=n))
paulis = [''.join(p) for p in paulis]

learnable = []

edges = []
edges_all = []
edge_id = 0
edge_to_id = {}
edge_to_pauli = {}
for p_in in paulis:
    p_out = Pauli(p_in[::-1]).evolve(C.adjoint()).to_label()[-n:][::-1]
    edges_all.append((int(pattern(p_in),2),int(pattern(p_out),2)))
    if pattern(p_out) != pattern(p_in):
        edge = (int(pattern(p_in),2),int(pattern(p_out),2))
        if edge not in edges:
            edges.append(edge)
            edge_to_id[edge] = edge_id
            edge_id += 1
            edge_to_pauli[edge] = [p_in]
        else:
            edge_to_pauli[edge].append(p_in)
    else:
        learnable.append(p_in)

E = len(edges)
G = nx.DiGraph(edges)
cycles = list(nx.simple_cycles(G))
cycles.sort(key=lambda x:len(x))

cycle_basis = []
cycle_basis_vecs = []
for cycle in cycles:
    cycle_vec = [0 for i in range(E)]
    for i in range(len(cycle)):
        edge = (cycle[i],cycle[(i+1)%len(cycle)])
        cycle_vec[edge_to_id[edge]] = 1

    cycle_basis_vecs.append(cycle_vec)
    r = len(cycle_basis_vecs)
    if matrix_rank(np.array(cycle_basis_vecs)) == r:
        cycle_basis.append(cycle)
    else:
        cycle_basis_vecs.pop()

cycle_basis_all = []
for p in learnable:
    v = [0 for i in range(4**n)]
    v[pauli_to_int(p)] = 1
    cycle_basis_all.append(v)


for cycle in cycle_basis:
    cycle_vec = [0 for i in range(4**n)]
    for i in range(len(cycle)):
        edge = (cycle[i],cycle[(i+1)%len(cycle)])
        cycle_vec[pauli_to_int(edge_to_pauli[edge][0])] = 1
    cycle_basis_all.append(cycle_vec)
    for i in range(len(cycle)):
        edge = (cycle[i],cycle[(i+1)%len(cycle)])
        paulis = edge_to_pauli[edge]
        if len(paulis) > 1:
            for p in paulis[1:]:
                cycle_vec_temp = copy.deepcopy(cycle_vec)
                cycle_vec_temp[pauli_to_int(paulis[0])] = 0
                cycle_vec_temp[pauli_to_int(p)] = 1
                cycle_basis_all.append(cycle_vec_temp)
                if matrix_rank(np.array(cycle_basis_all)) != len(cycle_basis_all):
                    cycle_basis_all.pop()

assert matrix_rank(np.array(cycle_basis_all)) == len(cycle_basis_all)
assert len(cycle_basis_all) == 4**n - 2**n + nx.number_strongly_connected_components(nx.DiGraph(edges_all))

C_dim = len(cycle_basis_all)

print("learnable Pauli errors:")
for cycle in cycle_basis_all:
    v_cycle = [0 for i in range(4**n)]
    s = ''
    for i in range(4**n):
        if cycle[i] == 1:
            pauli = int_to_pauli(n,i)
            s += pauli + ' '
            v = hadamard(pauli)
            for j in range(4**n):
                v_cycle[j] += v[j]
    cycle_basis_all.append(v_cycle)
    assert matrix_rank(np.array(cycle_basis_all)) == C_dim
    print(s)
    cycle_basis_all.pop()
print("all cycle errors are learnable!")