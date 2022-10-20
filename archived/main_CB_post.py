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
import CB_submit, CB_process
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

# True: efficient CB (simultaneously tackle with a stabilizer group), but cannot deal with degeneracy.
# stabilizer_group_cb = True
intercept_cb = False


n = 2
n_total = n
periodic = True
eps = 0.025
# eps_amp = 0.01
eps_readout = 0.01
# eps_cross = 0.01 # ENR/q=0.005
batch = 1

noise_model = NoiseModel()
# noise_model.add_all_qubit_quantum_error(pauli_error([('I',1-eps),('X',eps)]),['id', 'rz', 'sx'])
noise_model.add_all_qubit_quantum_error(pauli_error([('II',(1-eps)**2),('XX',eps**2),('IX',eps*(1-eps)),('XI',eps*(1-eps))]),['cx'])

# # phase/amplitude damping error
# amp_noise_1q = amplitude_damping_error(eps_amp)
# noise_model.add_all_qubit_quantum_error(amp_noise_1q.tensor(amp_noise_1q),['cx'])

# # crosstalk between CNOTs (choose it to be ZZ)
# crosstalk_noise = pauli_error([('II',1-eps_cross),('ZZ',eps_cross)])
# for i in range(n-2):
#     noise_model.add_nonlocal_quantum_error(crosstalk_noise, ['cx'], [i,i+1], [i+1,i+2])
# noise_model.add_nonlocal_quantum_error(crosstalk_noise, ['cx'], [n-2,n-1], [n-1,0])
# noise_model.add_nonlocal_quantum_error(crosstalk_noise, ['cx'], [n-1,0], [0,1])

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


# pauli_request_list = ['ZZZZ','IZZZ','IZIZ','IIIZ']
# pauli_sample_list = ['ZZZZ']


# pauli_sample_list = ['ZZZZ','XXXX','YYYY','ZYXZ']
# pauli_request_list = ['ZZZZ','IZZZ','IZIZ','IIIZ','XXXX','YYYY','ZYXZ']


fidelity_list = {} 
stdev_list = {}
if intercept_cb is True:
    intercept_list = {}
    intercept_std_list = {}

for pauli_sample in pauli_sample_list:

    with open("data"+pauli_sample+".txt", "r") as file_data:
        cb_data = json.load(file_data)
        
    n = cb_data["parameters"]['n']
    n_total = cb_data["parameters"]['n_total']
    shots = cb_data["parameters"]['shots']
    Lrange = cb_data["parameters"]['Lrange']
    C = cb_data["parameters"]['C']
    eps_readout = cb_data["parameters"]['eps_readout']
    repeat = cb_data["parameters"]['repeat']


    '''Add a dim to fidelity_list'''

    '''Add pauli request to speed up'''


    cb_result = CB_process.process_CB(n, batch, Lrange, cb_data, pauli_sample = pauli_sample, pauli_request_set = set(pauli_request_list),repeat=repeat, periodic=periodic,use_density_matrix=use_density_matrix,intercept_cb=intercept_cb)
    raw_fidelity_list = cb_result["fidelity_list"]

    # print(fidelity_list)
    # print('average fidelity = %f' % np.average(fidelity_list))

    # print(raw_fidelity_list)


    with open(filename, 'a') as f:
        f.write(str(raw_fidelity_list)+'\n')
    
    # if use_ibmq:
    #     with open('ibmq_cb.txt', 'a') as f:
    #         f.write(str(cb_data))
    # else:
    #     with open('test_cb.txt','a') as f:
    #         f.write(str(cb_data))

    for sub_label in raw_fidelity_list.keys():
        if sub_label in fidelity_list:
            continue
        elif(sub_label == 'I'*n):
            fidelity_list[sub_label] = 1.0
            stdev_list[sub_label] = 0.0
            if intercept_cb is True:
                intercept_list[sub_label] = 1.0
                intercept_std_list[sub_label] = 0.0

        else:
            if intercept_cb is False:
                alpha, alpha_err = CB_process.fit_CB(Lrange, raw_fidelity_list[sub_label])
                # print("CB for %dth Pauli:" % k, alpha, alpha_err)
                fidelity_list[sub_label] = alpha
                stdev_list[sub_label] = alpha_err
            else:
                alpha, alpha_err, intercept, intercept_err = CB_process.fit_CB_2(Lrange, raw_fidelity_list[sub_label])
                # print("CB for %dth Pauli:" % k, alpha, alpha_err)
                fidelity_list[sub_label] = alpha
                stdev_list[sub_label] = alpha_err
                intercept_list[sub_label] = intercept
                intercept_std_list[sub_label] = intercept_err

# sys.exit(0)

# print("\nFinal result:")
# print('n =', n, ' K =', K, ' C =', C, ' Lrange =', list(Lrange))
print("Parameters: n = %d, C = %d, " % (n,C), "L = ", str(Lrange))

if use_density_matrix:
    print("Density matrix based simulation")
else:
    print("Measurement based simulation, shots = %d" % shots)

if use_readout_error is True:
    print("Measurement bitflip rate = %f" % eps_readout)
else:
    print("No readout error")

if intercept_cb is False:

    print("Label / Pauli infidelity / Standard deviation")
    for pauli_label in pauli_request_list:
        # print(pauli_label, 1-fidelity_list[pauli_label], stdev_list[pauli_label])
        print(pauli_label+" %.6f, %.6f"%(1-fidelity_list[pauli_label], stdev_list[pauli_label]))
    # print('Effective noise rate = ' + str(1-np.average(list(fidelity_list.values()))))



    with open(filename, 'a') as f:
        f.write("Parameters: n = %d, C = %d, " % (n,C) + "L = " + str(Lrange) + '\n')
        # f.write(str(raw_fidelity_list)+'\n')
        f.write("Label / Pauli infidelity / Standard deviation\n")
        for pauli_label in pauli_request_list:
            f.write(str(pauli_label) + ', ' + str(1-fidelity_list[pauli_label]) + ', ' + str(stdev_list[pauli_label])+'\n')
        f.write('Effective noise rate = ' + str(1-np.average(list(fidelity_list.values()))) +'\n')
        f.write("-------------\n")

else:

    print("Label / Pauli infidelity / Std(fidelity) / Intercept / Std(intercept)")
    for pauli_label in pauli_request_list:
        print(pauli_label, 1-fidelity_list[pauli_label], stdev_list[pauli_label], intercept_list[pauli_label], intercept_std_list[pauli_label])
    print('Effective noise rate = ' + str(1-np.average(list(fidelity_list.values()))))



    with open(filename, 'a') as f:
        # f.write(str(raw_fidelity_list)+'\n')
        f.write("Parameters: n = %d, C = %d, " % (n,C) + "L = " + str(Lrange) + '\n')
        f.write("Label / Pauli infidelity / Std(fidelity) / Intercept / Std(intercept)\n")
        for pauli_label in pauli_request_list:
            f.write(str(pauli_label)+', '+str(1-fidelity_list[pauli_label]) + ', ' +str(stdev_list[pauli_label]) + ', ' + str(intercept_list[pauli_label]) +', ' + str(intercept_std_list[pauli_label])+'\n')
        f.write('Effective noise rate = ' + str(1-np.average(list(fidelity_list.values())))+'\n')
        f.write("-------------\n")
