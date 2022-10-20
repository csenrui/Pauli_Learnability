import numpy as np
import sys, json, copy, time, pickle, random
from concurrent.futures import ThreadPoolExecutor
import qiskit
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.aer import StatevectorSimulator, AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error, amplitude_damping_error, ReadoutError
from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet
from qiskit.providers.ibmq.apiconstants import ApiJobShareLevel
from qiskit.qobj.utils import MeasLevel
import qiskit.quantum_info as qi
from qubit_map import qubit_maps
import matplotlib.pyplot as plt
from scipy.stats import sem, unitary_group
from scipy import sparse
from statistics import stdev
import itertools
from qiskit.compiler import transpile
import CB_submit, CB_corr_submit


shots = 1000
shots_intc = 2000
n = 2
n_total = n
Leven = [2**x for x in range(1,9)]
Lodd = [(2**x)+1 for x in range(1,9)]
repeat_even = [1 for k in Leven] # not-used
repeat_odd = [1 for k in Lodd] # not-used
C = 60
C_intc = 150
batch = 1
gset = "Pauli"
q = qubit_maps['local']



######## simulator or ibmq
use_simulator = False


# input ibmq credential
IBMQ.load_account()
provider = IBMQ.enable_account("...",hub='...', group='...', project='...')
backend = provider.get_backend('ibmq_montreal')

delay_time = backend.properties().gate_length('sx',qubits=q[1]) # in second
print("delay_time = ",str(delay_time))


print(backend.properties().gate_length('sx',qubits=q[1]))

print(backend.properties().gate_length('x',qubits=q[1]))


### if no delay for single-qubit layer
# delay_time = 0
# print("no delay")
# print(delay_time)


if use_simulator is True:
    filename = 'simulation_all_20220208'
    noise_model = NoiseModel.from_backend(provider.get_backend('ibm_perth'))
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
else:
    filename = 'ibmq_experiment_all_20220323'
    job_manager = IBMQJobManager()



print("CB, ", "n=%d" % n)
print("simulation method:", backend.configuration().description)

data = {}
token = ''.join(random.choice([str(j) for j in range(10)]) for i in range(10))
data['token'] = token
filename += '_' + token

####################### experiment1: standard CB
data['cb'] = {}
pauli_sample_list = [''.join(s) for s in itertools.product(['X','Y','Z'], repeat = n)]
# pauli_sample_list = ['XZ','ZX']

for pauli_sample in pauli_sample_list:
    tag = 'cb_' + pauli_sample
    print(tag)
    Lrange = Leven
    repeat = repeat_even #not used


    cb_data, cb_circ_all = CB_submit.submit_cb(n,n_total,Lrange=Lrange,C=C,batch=batch, pauliList = pauli_sample, qubit_map=q,gset=gset,repeat=repeat,periodic=False,use_density_matrix=False,use_reset_error = use_simulator,delay_time=delay_time)

    # print(transpile(cb_circ_all[0][0],backend=backend,initial_layout=[0,1],optimization_level=2))
    # sys.exit(0)

    print("created %d circuits" % len(cb_circ_all[0]))



    if use_simulator is True:
        job = backend.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[0,1],optimization_level=2), shots=shots_intc, max_parallel_experiments=0, memory = True) 
        result = job.result()
    else:
        job_set = job_manager.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[0,1],optimization_level=2), shots=shots, memory=True, backend=backend, name='cb', job_tags=[tag,token])
        job_set_id = job_set.job_set_id()

    cb_data["parameters"] = {}
    cb_data["parameters"]['n'] = n 
    cb_data["parameters"]['n_total'] = n_total
    cb_data["parameters"]['shots'] = shots 
    cb_data["parameters"]['Lrange'] = Lrange 
    cb_data["parameters"]['C'] = C 
    cb_data["parameters"]['repeat'] = repeat
    cb_data["parameters"]['pauli'] = pauli_sample

    if use_simulator is True:
        cb_data["result"] = [result]
    else:
        cb_data["job_set_id"] = job_set_id


    
    data['cb'][tag] = cb_data


    

####################### experiment2: interleaved CB
data['int_cb'] = {}
pauli_sample_list = [''.join(s) for s in itertools.product(['X','Y','Z'], repeat = n)]
# pauli_sample_list = ['XZ','ZX']

for pauli_sample in pauli_sample_list:
    tag = 'int_cb_' + pauli_sample
    print(tag)
    Lrange = Leven
    repeat = repeat_even #not used
    cb_data, cb_circ_all = CB_corr_submit.submit_cb(n,n_total,Lrange=Lrange,C=C,batch=batch, pauliList = pauli_sample, qubit_map=q,gset=gset,repeat=repeat,periodic=False,use_density_matrix=False,use_reset_error = use_simulator)
    print("created %d circuits" % len(cb_circ_all[0]))


    if use_simulator is True:
        job = backend.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[0,1],optimization_level=2), shots=shots_intc, max_parallel_experiments=0, memory = True)  
        result = job.result()
    else:
        job_set = job_manager.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[0,1],optimization_level=2), shots=shots, memory=True, backend=backend, name='int_cb', job_tags=[tag,token])
        job_set_id = job_set.job_set_id()    



    cb_data["parameters"] = {}
    cb_data["parameters"]['n'] = n 
    cb_data["parameters"]['n_total'] = n_total
    cb_data["parameters"]['shots'] = shots 
    cb_data["parameters"]['Lrange'] = Lrange 
    cb_data["parameters"]['C'] = C 
    cb_data["parameters"]['repeat'] = repeat
    cb_data["parameters"]['pauli'] = pauli_sample
    
    if use_simulator is True:
        cb_data["result"] = [result]
    else:
        cb_data["job_set_id"] = job_set_id

    data['int_cb'][tag] = cb_data



####################### experiment3: intercept CB
data['intc_cb'] = {}
parity_pauli_sample_list = [('XX',0),('IX',1),('ZZ',0),('ZI',1)] # Only 2 degrees of freedom
# parity_pauli_sample_list = [('XX',0),('IX',1)]

for pauli_sample, parity in parity_pauli_sample_list:
    tag = 'intc_cb_' + pauli_sample
    print(tag)

    Lrange = (Leven,Lodd)[parity]
    repeat = (repeat_even,repeat_odd)[parity] #not used

    cb_data, cb_circ_all = CB_submit.submit_cb(n,n_total,Lrange=Lrange,C=C_intc,batch=batch, pauliList = pauli_sample, qubit_map=q,gset=gset,repeat=repeat,periodic=False,use_density_matrix=False,use_reset_error = use_simulator)
    print("created %d circuits" % len(cb_circ_all[0]))

    if use_simulator is True:
        job = backend.run(cb_circ_all[0], shots=shots_intc, max_parallel_experiments=0, memory = True) 
        result = job.result()
    else:
        job_set = job_manager.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[0,1]), shots=shots_intc, memory=True, backend=backend, name='intc_cb', job_tags=[tag,token])
        job_set_id = job_set.job_set_id()    

    cb_data["parameters"] = {}
    cb_data["parameters"]['n'] = n 
    cb_data["parameters"]['n_total'] = n_total
    cb_data["parameters"]['shots'] = shots_intc
    cb_data["parameters"]['Lrange'] = Lrange 
    cb_data["parameters"]['C'] = C_intc 
    cb_data["parameters"]['repeat'] = repeat
    cb_data["parameters"]['pauli'] = pauli_sample
    cb_data["parameters"]['parity'] = parity

    if use_simulator is True:
        cb_data["result"] = [result]
    else:
        cb_data["job_set_id"] = job_set_id

    data['intc_cb'][tag] = cb_data



####################### save data

if use_simulator is True:
    with open('data/' + filename + '_full', 'wb') as outfile:
        pickle.dump(data, outfile)
else:
    with open('data/' + filename, 'wb') as outfile:
        pickle.dump(data, outfile)

print(token)