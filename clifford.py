import numpy as np
import itertools, copy, random, sys
import qiskit
from qiskit import quantum_info
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliList

n = 4
depth = 2**n-1
# depth=2

CNOT = qiskit.circuit.library.CXGate()
SWAP = qiskit.circuit.library.SwapGate()
# C = quantum_info.Clifford.from_circuit(CNOT)
# C = quantum_info.Clifford.from_circuit(SWAP)

qc = QuantumCircuit(4)
qc.cx(0, 1)
qc.cx(2, 3)
C = quantum_info.Clifford.from_circuit(qc)
# print(C)
# sys.exit(0)

# C = quantum_info.random_clifford(n,seed=100)

print(C.to_circuit())

paulis = list(itertools.product(['I','X','Y','Z'], repeat=n))
paulis = [''.join(p) for p in paulis]
# pauli_list = PauliList(paulis)

evolve_dict = {}
pattern_dict = {}
# print(pauli_list)

# invariant = []
learnable = []
unlearnable = []
weight_transfer = []

def pattern(pauli):
    pattern = ''
    for i in range(len(pauli)):
        if pauli[i] == 'I':
            pattern += '0'
        else:
            pattern += '1'
    return pattern


# invariant points
for p_in in paulis:
    p_out = Pauli(p_in).evolve(C).to_label()[-n:]
    evolve_dict[p_in] = p_out
    pattern_dict[p_in] = pattern(p_in)
    # print(p_in, p_out)
    # if p_out.equiv(p_in):
    #     invariant.append(p_in.to_label())
    if pattern(p_out) == pattern(p_in):
        learnable.append(p_in)
    if pattern(p_out) != pattern(p_in):
        # print(p_in, p_out)
        unlearnable.append(p_in)
        t = (pattern(p_in),pattern(p_out))
        if t not in weight_transfer:
            weight_transfer.append(t)

# print(evolve_dict)
# print(pattern_dict)
# sys.exit(0)

# unlearnable_copy = copy.deepcopy(unlearnable)
# learnable_copy = copy.deepcopy(learnable)
# unlearnable_freedom = 0
# unlearnable_group = []

# while True:
#     if len(unlearnable_copy) == 0:
#         break
#     # p_label = random.choice(unlearnable_copy)
#     p_label = unlearnable_copy[0]
#     group = []
#     group.append(p_label)
#     # print(p_label,1)
#     unlearnable_freedom += 1
#     learnable_copy.append(p_label)
#     unlearnable_copy.remove(p_label)
#     while True:
#         keep = False
#         for q_label in copy.deepcopy(unlearnable_copy):
#             # print(q_label,3)
#             p_in_1 = Pauli(q_label)
#             p_out_1 = p_in_1.evolve(C)
#             for p_in_2_label in learnable_copy:
#                 p_in_2 = Pauli(p_in_2_label)
#                 if pattern(n,p_in_2) == pattern(n,p_out_1):
#                     p_out_2 = p_in_2.evolve(C)
#                     if pattern(n,p_out_2) == pattern(n,p_in_1):
#                         # print(q_label,2)
#                         learnable_copy.append(q_label)
#                         # print(unlearnable_copy)
#                         if q_label in unlearnable_copy:
#                             unlearnable_copy.remove(q_label)
#                             group.append(q_label)
#                         keep = True
#         if keep == False:
#             unlearnable_group.append(group)
#             break
    
    


# # # print("invariant points:", invariant)
# # print("weight transfer:", weight_transfer, len(weight_transfer))
# # for t in weight_transfer:
# #     p,q = t 
# #     if (q,p) not in weight_transfer:
# #         print("asymmetric!")
# print("learnable Pauli fidelities:", learnable)
# print("unlearnable Pauli fidelities:", unlearnable)
# print("unlearnable degrees of freedom:", unlearnable_freedom)
# print("unlearnable groups:", unlearnable_group)

# # print(unlearnable_freedom == int(len(weight_transfer)/2))

unlearnable_copy = copy.deepcopy(unlearnable)
learnable_copy = []
unlearnable_freedom = 0
unlearnable_group = []

while True:
    if len(unlearnable_copy) == 0:
        break
    # p_label = random.choice(unlearnable_copy)
    p_label = unlearnable_copy[0]
    group = []
    group.append(p_label)
    # print(p_label,1)
    unlearnable_freedom += 1
    learnable_copy.append(p_label)
    unlearnable_copy.remove(p_label)
    while True:
        keep = False
        for q_label in copy.deepcopy(unlearnable_copy):
            # print("here", q_label, len(learnable_copy))
            # print(q_label,3)
            p_in_1 = q_label
            pattern_in = pattern_dict[p_in_1]
            p_out_1 = evolve_dict[p_in_1]
            reachable = [pattern_dict[p_out_1]]
            # print(reachable)
            learned = False
            for i in range(depth-1):
                for p_in_2 in learnable_copy:
                    if pattern_dict[p_in_2] in reachable:
                        p_out_2 = evolve_dict[p_in_2]
                        # print(p_in_2,p_out_2)
                        reachable.append(pattern_dict[p_out_2])
                        if pattern_dict[p_out_2] == pattern_in:
                            learned = True
                            break
                if learned:
                    break
            # print(reachable)
            # sys.exit(0)
            if pattern_in in reachable:
                learnable_copy.append(q_label)
                # print(unlearnable_copy)
                if q_label in unlearnable_copy:
                    unlearnable_copy.remove(q_label)
                    group.append(q_label)
                keep = True
        if keep == False:
            unlearnable_group.append(group)
            break
    
    


# # print("invariant points:", invariant)
# print("weight transfer:", weight_transfer, len(weight_transfer))
# for t in weight_transfer:
#     p,q = t 
#     if (q,p) not in weight_transfer:
#         print("asymmetric!")
print("learnable Pauli fidelities:", learnable)
print("unlearnable Pauli fidelities:", unlearnable)
print("unlearnable degrees of freedom:", unlearnable_freedom)
print("unlearnable groups:", unlearnable_group)

# print(unlearnable_freedom == int(len(weight_transfer)/2))