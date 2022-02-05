import numpy as np
import sys, json, copy, time, pickle, random
from concurrent.futures import ThreadPoolExecutor
import qiskit
from qiskit import IBMQ, QuantumCircuit, execute
# from qiskit.providers.aer import StatevectorSimulator, AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error, amplitude_damping_error, ReadoutError
import qiskit.ignis.verification.randomized_benchmarking as rb
# from qiskit.providers.aer.noise.errors.errorutils import single_qubit_clifford_gates
# from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet
# sfrom qiskit.providers.ibmq.apiconstants import ApiJobShareLevel
from qiskit.qobj.utils import MeasLevel
from sympy import N
from qubit_map import qubit_maps
import matplotlib.pyplot as plt
from scipy.stats import sem, unitary_group
# from scipy import sparse
import CB_process
from statistics import stdev
import itertools
# from qiskit.compiler import transpile

#Change test

use_density_matrix = False # density matrix based / measurement based simulation


filename_label = 'ibmq_experiment_all_20220131_3792580001'
#filename_label = 'ibmq_experiment_all_20220131_3975437641'

with open('data/' + filename_label + '_full', 'rb') as infile:
    data = pickle.load(infile)
token = data["token"]



result = {}

# ############# exp 1: CB
cb = data['cb']

fidelity_list = {} 
stdev_list = {}
for tag in cb:
    print(tag)
    cb_data = cb[tag]

    n = cb_data["parameters"]['n']
    n_total = cb_data["parameters"]['n_total']
    shots = cb_data["parameters"]['shots']
    Lrange = cb_data["parameters"]['Lrange']
    C = cb_data["parameters"]['C']
    repeat = cb_data["parameters"]['repeat']
    pauli_sample = cb_data["parameters"]['pauli']

    cb_result = CB_process.process_CB(n, C, shots, 1, Lrange, cb_data, pauli_sample = pauli_sample,repeat=repeat, periodic=True,use_density_matrix=use_density_matrix,intercept_cb=False)
    raw_fidelity_list = cb_result["fidelity_list"]


    for sub_label in raw_fidelity_list.keys():
        if sub_label in fidelity_list:
            continue # wasteful!
        elif(sub_label == 'I'*n):
            fidelity_list[sub_label] = 1.0
            stdev_list[sub_label] = 0.0
        else:
            alpha, alpha_err = CB_process.fit_CB(Lrange, raw_fidelity_list[sub_label])
            fidelity_list[sub_label] = alpha
            stdev_list[sub_label] = alpha_err

print("std_CB")
print("Parameters: n = %d, C = %d, " % (n,C), "L = ", str(Lrange))
print("Label / Pauli infidelity / Standard deviation")
pauli_request_list = [''.join(s) for s in itertools.product(['I','X','Y','Z'], repeat = n)]
for pauli_label in pauli_request_list:
    print(pauli_label+" %.6f, %.6f"%(1-fidelity_list[pauli_label], stdev_list[pauli_label]))

result['cb'] = {
    "fidelity": fidelity_list,
    "stdev":    stdev_list,
    "n":        n,
    "n_total":  n_total,
    "Lrange":   Lrange,
    "C":        C,
    "batch":    1,
    "repeat":   repeat
}

############# exp 2: int_CB
cb = data['int_cb']

fidelity_list = {} 
stdev_list = {}
for tag in cb:
    print(tag)
    cb_data = cb[tag]

    n = cb_data["parameters"]['n']
    n_total = cb_data["parameters"]['n_total']
    shots = cb_data["parameters"]['shots']
    Lrange = cb_data["parameters"]['Lrange']
    C = cb_data["parameters"]['C']
    repeat = cb_data["parameters"]['repeat']
    pauli_sample = cb_data["parameters"]['pauli']

    cb_result = CB_process.process_CB(n, C, shots, 1, Lrange, cb_data, pauli_sample = pauli_sample,repeat=repeat, periodic=True,use_density_matrix=use_density_matrix,intercept_cb=False)
    raw_fidelity_list = cb_result["fidelity_list"]


    for sub_label in raw_fidelity_list.keys():
        if sub_label in fidelity_list:
            continue # wasteful!
        elif(sub_label == 'I'*n):
            fidelity_list[sub_label] = 1.0
            stdev_list[sub_label] = 0.0
        else:
            alpha, alpha_err = CB_process.fit_CB(Lrange, raw_fidelity_list[sub_label])
            fidelity_list[sub_label] = alpha
            stdev_list[sub_label] = alpha_err

print("int_CB")
print("Parameters: n = %d, C = %d, " % (n,C), "L = ", str(Lrange))
print("Label / Pauli infidelity / Standard deviation")
pauli_request_list = [''.join(s) for s in itertools.product(['I','X','Y','Z'], repeat = n)]
for pauli_label in pauli_request_list:
    print(pauli_label+" %.6f, %.6f"%(1-fidelity_list[pauli_label], stdev_list[pauli_label]))


result['int_cb'] = {
    "fidelity": fidelity_list,
    "stdev":    stdev_list,
    "n":        n,
    "n_total":  n_total,
    "Lrange":   Lrange,
    "C":        C,
    "batch":    1,
    "repeat":   repeat
}


############# exp 3: intc_CB

parity_pauli_sample_list = [('XX',0),('IX',1),('ZZ',0),('ZI',1)]
cb = data['intc_cb']

use_boostrap = False
C_max = 0
shots_max = 0

# fidelity_list = {} 
# stdev_list = {}
# intercept_list = {}
# intercept_std_list = {}
# covar_list = {}
params_list = {}
pcov_list = {}

for tag in cb:
    print(tag)
    cb_data = cb[tag]
    n = cb_data["parameters"]['n']
    n_total = cb_data["parameters"]['n_total']
    shots = cb_data["parameters"]['shots']
    Lrange = cb_data["parameters"]['Lrange']
    C = cb_data["parameters"]['C']
    repeat = cb_data["parameters"]['repeat']
    pauli_sample = cb_data["parameters"]['pauli']
    parity = cb_data["parameters"]['parity']

    cb_result = CB_process.process_CB(n, C, shots, 1, Lrange, cb_data, pauli_sample = pauli_sample,repeat=repeat, periodic=True,use_density_matrix=use_density_matrix,intercept_cb=True,C_max=C_max,shots_max=shots_max,use_boostrap=use_boostrap)
    raw_fidelity_list = cb_result["fidelity_list"]


    for sub_label in raw_fidelity_list.keys():
        if (sub_label,parity) in params_list:
            continue
        elif(sub_label == 'I'*n):
            params_list = np.array([1.0,1.0])
            pcov_list = np.array([[0.0,0.0],[0.0,0.0]])
        else:
            params, pcov = CB_process.fit_CB_all(Lrange, raw_fidelity_list[sub_label])
            params_list[(sub_label,parity)] = params
            pcov_list[(sub_label,parity)] = pcov
            # note: parameters = (intercept, slope)

noise_after_the_gate = True
dec_fidelity_list = {}
dec_stdev_list = {}
for k in range(0,len(parity_pauli_sample_list),2):
    pp1 = parity_pauli_sample_list[k]
    pp2 = parity_pauli_sample_list[k+1]
    # f = (fidelity_list[pp1]+fidelity_list[pp2])/2
    # r = (intercept_list[pp2]/intercept_list[pp1])
    f = (params_list[pp1][1] + params_list[pp2][1])/2
    r = (params_list[pp2][0]/params_list[pp1][0])
    if noise_after_the_gate is True:
        lambda1 = f * r
        lambda2 = f / r
    else:
        lambda1 = f / r
        lambda2 = f * r
    # propagation of uncertainty
    s1, s2 =  CB_process.calculate_uncertainty(params_list[pp1],params_list[pp2],pcov_list[pp1],pcov_list[pp1])

    dec_fidelity_list[pp1[0]] = lambda1
    dec_fidelity_list[pp2[0]] = lambda2
    dec_stdev_list[pp1[0]] = s1
    dec_stdev_list[pp2[0]] = s2

print("Label / Pauli infidelity / Std(fidelity) / Intercept / Std(intercept)")
for pauli_label in parity_pauli_sample_list:
    print(str(pauli_label)+" %.5f %.5f %.5f %.5f"%(1-params_list[pauli_label][1], np.sqrt(pcov_list[pauli_label][1,1]), 1-params_list[pauli_label][0], np.sqrt(pcov_list[pauli_label][0,0])))


print("Label / Pauli infidelity (decoupled)/ Std")
for pauli_label in parity_pauli_sample_list:
    print(str(pauli_label[0])+" %.5f %.5f"%(1-dec_fidelity_list[pauli_label[0]],dec_stdev_list[pauli_label[0]]))

result['intc_cb'] = {
    "params":       params_list,
    "pcov":         pcov_list,
    "dec_fidelity": dec_fidelity_list,
    "dec_stdev":    dec_stdev_list,
    "n":            n,
    "n_total":      n_total,
    "Lrange":       Lrange,
    "C":            C,
    "batch":        1,
    "repeat":       repeat,
    "parity_pauli_sample_list": parity_pauli_sample_list
}

filename = 'data/' + filename_label + '_result_var'
with open(filename, 'wb') as outfile:
    pickle.dump(result, outfile)
