import numpy as np
import sys, json, copy, time, pickle, random
from concurrent.futures import ThreadPoolExecutor
import qiskit
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.aer import StatevectorSimulator, AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error, amplitude_damping_error, ReadoutError
import qiskit.ignis.verification.randomized_benchmarking as rb
from qiskit.providers.aer.noise.errors.errorutils import single_qubit_clifford_gates
from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet
from qiskit.providers.ibmq.apiconstants import ApiJobShareLevel
from qiskit.qobj.utils import MeasLevel
from qubit_map import qubit_maps
import matplotlib.pyplot as plt
from scipy.stats import sem, unitary_group
from scipy import sparse
import CB_corr_submit as CB_submit
#import CB_submit
from statistics import stdev
import itertools
from qiskit.compiler import transpile

#Change test

use_density_matrix = False # density matrix based / measurement based simulation




#### Set one and only one of the following to be true:
use_ibmq = False # run on ibm devices
use_stabilizer_simulator = False # whether stabilizer simulator is used (valid only for Pauli noise)
use_density_matrix_sample = True # use density matrix simulation, but returns samples
use_state_vector_sample = False # use state vector simulation, which returns samples

if use_ibmq:
    filename = 'ibmq_cb_single.txt'
else:
    filename = 'simulate_cb_single.txt'


use_readout_error = True
use_reset_error = True


# True: efficient CB (simultaneously tackle with a stabilizer group), but cannot deal with degeneracy.
# stabilizer_group_cb = True

shots = 100 # for measurement-based simulation only
n = 2
n_total = n
Lrange = list(range(2,39,4)) # len = 10
# Lrange = list(range(3,40,4))
# Lrange = [4]
repeat = [1 for k in Lrange] # not-used
C = 25
#C = 1
batch = 1
gset = "Pauli"
# K = 25
q = qubit_maps['local']
# q = {
#     0:1,
#     1:2,
#     2:3,
#     3:4,
#     4:0
# }
# basis_gates = ['id', 'rz', 'sx', 'cx']
periodic = True
eps = 0.05
# eps_amp = 0.01
eps_readout = 0.01
# eps_reset = 0.01
# eps_cross = 0.01 # ENR/q=0.005

noise_model = NoiseModel()
# noise_model.add_all_qubit_quantum_error(pauli_error([('I',1-eps),('X',eps)]),['id', 'rz', 'sx'])
# noise_model.add_all_qubit_quantum_error(pauli_error([('II',(1-eps)**2),('XX',eps**2),('IX',eps*(1-eps)),('XI',eps*(1-eps))]),['cx'])

# # phase/amplitude damping error
amp_noise_1q = amplitude_damping_error(eps)
noise_model.add_all_qubit_quantum_error(amp_noise_1q.tensor(amp_noise_1q),['cx'])

# # crosstalk between CNOTs (choose it to be ZZ)
# crosstalk_noise = pauli_error([('II',1-eps_cross),('ZZ',eps_cross)])
# for i in range(n-2):
#     noise_model.add_nonlocal_quantum_error(crosstalk_noise, ['cx'], [i,i+1], [i+1,i+2])
# noise_model.add_nonlocal_quantum_error(crosstalk_noise, ['cx'], [n-2,n-1], [n-1,0])
# noise_model.add_nonlocal_quantum_error(crosstalk_noise, ['cx'], [n-1,0], [0,1])


# reset error
# if use_reset_error is True:
#     pass

# readout error
if use_readout_error is True:
    p0given1 = eps_readout
    p1given0 = eps_readout
    readout_noise_1q = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
    noise_model.add_all_qubit_readout_error(readout_noise_1q)


if use_density_matrix or use_density_matrix_sample:
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
elif use_state_vector_sample:
    backend = AerSimulator(method='statevector', noise_model=noise_model)
elif use_stabilizer_simulator:
    backend = AerSimulator(method='stabilizer', noise_model=noise_model)
elif use_ibmq:
    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmq_manila')
else:
    assert 1==0

print("CB, ", "n=%d" % n)
print("simulation method:", backend.configuration().description)



##Pauli sample choose here!

'''Pauli request'''
pauli_request_list = [''.join(s) for s in itertools.product(['I','X','Y','Z'], repeat = n)]
pauli_sample_list = [''.join(s) for s in itertools.product(['X','Y','Z'], repeat = n)]

# pauli_sample_list = pauli_request_list # for single-Pauli CB

'''For verification on 5-qubit IBM device'''
# pauli_request_list = ['II','XI','IY','XY']
# pauli_sample_list = ['XY']

'''Single Pauli CB'''
# pauli_request_list = ['ZI','IX','YI','IY','ZX','YX']
# pauli_request_list = ['ZZ','XX','YZ','XY','YY','ZY']
# pauli_sample_list = pauli_request_list

# pauli_request_list = ['ZIII','ZZII','IZZZ','ZZZZ']
# pauli_sample_list = ['ZZZZ']

# pauli_sample_list = ['ZZZZ','XXXX','YYYY','ZYXZ']



# fidelity_list = {} 
# stdev_list = {}
# if stabilizer_group_cb is False:
#     intercept_list = {}
#     intercept_std_list = {}

for pauli_sample in pauli_sample_list:

    cb_data, cb_circ_all = CB_submit.submit_cb(n,n_total,Lrange=Lrange,C=C,batch=batch, pauliList = pauli_sample, qubit_map=q,gset=gset,repeat=repeat,periodic=periodic,use_density_matrix=use_density_matrix)
    # print("created %d circuits" % len(cb_circ_all[0]))

    print(cb_circ_all[0][0])

    #print(transpile(cb_circ_all[0][0],basis_gates=backend.configuration().basis_gates))

    #print(transpile(cb_circ_all[0][0],backend=backend,initial_layout=[1,2]))


    if use_density_matrix is True:
        job = backend.run(cb_circ_all[0], shots=1, max_parallel_experiments=0) 
    elif use_ibmq is False:
        job = backend.run(cb_circ_all[0], shots=shots, max_parallel_experiments=0, memory = True) 
    else:    
        job = backend.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[1,2]), shots=shots, memory = True) 
    result = job.result()

    cb_data["result"] = [result.to_dict()]

    cb_data["parameters"] = {}
    cb_data["parameters"]['n'] = n 
    cb_data["parameters"]['n_total'] = n_total
    cb_data["parameters"]['shots'] = shots 
    cb_data["parameters"]['Lrange'] = Lrange 
    cb_data["parameters"]['C'] = C 
    cb_data["parameters"]['eps_readout'] = eps_readout
    cb_data["parameters"]['repeat'] = repeat


    # test: data saving

    # print(cb_data)

    with open("data"+ pauli_sample + ".txt", "w") as file_data:
        json.dump(cb_data,file_data)

