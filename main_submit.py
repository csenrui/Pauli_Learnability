import numpy as np
import sys, json, copy, time, pickle, random
from concurrent.futures import ThreadPoolExecutor
import qiskit
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.aer import StatevectorSimulator, AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error, amplitude_damping_error, ReadoutError
from qiskit.providers.aer.noise.errors.errorutils import single_qubit_clifford_gates
from qiskit.providers.ibmq.managed import IBMQJobManager, ManagedJobSet
from qiskit.providers.ibmq.apiconstants import ApiJobShareLevel
from qiskit.qobj.utils import MeasLevel
from qubit_map import qubit_maps
import matplotlib.pyplot as plt
from scipy.stats import sem, unitary_group
from scipy import sparse
from statistics import stdev
import itertools
from qiskit.compiler import transpile
import CB_submit, CB_corr_submit

filename = 'ibmq_experiment_all_20220127'
shots = 1000 # for measurement-based simulation only
shots_intc = 8000
n = 2
n_total = n
#Lrange = list(range(2,39,4)) # len = 10
Leven = list(range(2,39,4))
Lodd = list(range(3,40,4))
#repeat = [1 for k in Lrange] # not-used
repeat_even = [1 for k in Leven] # not-used
repeat_odd = [1 for k in Lodd] # not-used
C = 30
C_intc = 90
#C = 1
batch = 1
gset = "Pauli"
q = qubit_maps['local']

IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-research', group='berkeley-6', project='main')
# provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='sys-reserve')
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy164')
# backend = provider.get_backend('ibmq_manila')
backend = provider.get_backend('ibm_perth')

print("CB, ", "n=%d" % n)
print("simulation method:", backend.configuration().description)

data = {}
job_manager = IBMQJobManager()
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

    cb_data, cb_circ_all = CB_submit.submit_cb(n,n_total,Lrange=Lrange,C=C,batch=batch, pauliList = pauli_sample, qubit_map=q,gset=gset,repeat=repeat,periodic=False,use_density_matrix=False)
    print("created %d circuits" % len(cb_circ_all[0]))

    # print(cb_circ_all[0][0])
    job_set = job_manager.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[1,2]), shots=shots, memory=True, backend=backend, name='cb', job_tags=[tag,token])
    job_set_id = job_set.job_set_id()    

    cb_data["parameters"] = {}
    cb_data["parameters"]['n'] = n 
    cb_data["parameters"]['n_total'] = n_total
    cb_data["parameters"]['shots'] = shots 
    cb_data["parameters"]['Lrange'] = Lrange 
    cb_data["parameters"]['C'] = C 
    cb_data["parameters"]['repeat'] = repeat
    cb_data["parameters"]['pauli'] = pauli_sample

    cb_data["job_set_id"] = job_set_id

    data['cb'][tag] = cb_data
    # test: data saving

    print(cb_data)

    

####################### experiment2: interleaved CB
data['int_cb'] = {}
pauli_sample_list = [''.join(s) for s in itertools.product(['X','Y','Z'], repeat = n)]
# pauli_sample_list = ['XZ','ZX']

for pauli_sample in pauli_sample_list:
    tag = 'int_cb_' + pauli_sample
    print(tag)
    Lrange = Leven
    repeat = repeat_even #not used

    cb_data, cb_circ_all = CB_corr_submit.submit_cb(n,n_total,Lrange=Lrange,C=C,batch=batch, pauliList = pauli_sample, qubit_map=q,gset=gset,repeat=repeat,periodic=False,use_density_matrix=False)
    print("created %d circuits" % len(cb_circ_all[0]))

    # print(cb_circ_all[0][0])
    job_set = job_manager.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[1,2]), shots=shots, memory=True, backend=backend, name='int_cb', job_tags=[tag,token])
    job_set_id = job_set.job_set_id()    

    cb_data["parameters"] = {}
    cb_data["parameters"]['n'] = n 
    cb_data["parameters"]['n_total'] = n_total
    cb_data["parameters"]['shots'] = shots 
    cb_data["parameters"]['Lrange'] = Lrange 
    cb_data["parameters"]['C'] = C 
    cb_data["parameters"]['repeat'] = repeat
    cb_data["parameters"]['pauli'] = pauli_sample

    cb_data["job_set_id"] = job_set_id

    data['int_cb'][tag] = cb_data
    # test: data saving

    print(cb_data)


####################### experiment3: intercept CB
data['intc_cb'] = {}
parity_pauli_sample_list = [('XX',0),('IX',1),('ZZ',0),('ZI',1)] # Only 2 degrees of freedom
# parity_pauli_sample_list = [('XX',0),('IX',1)]

for pauli_sample, parity in parity_pauli_sample_list:
    tag = 'intc_cb_' + pauli_sample
    print(tag)

    Lrange = (Leven,Lodd)[parity]
    repeat = (repeat_even,repeat_odd)[parity] #not used

    cb_data, cb_circ_all = CB_submit.submit_cb(n,n_total,Lrange=Lrange,C=C_intc,batch=batch, pauliList = pauli_sample, qubit_map=q,gset=gset,repeat=repeat,periodic=False,use_density_matrix=False)
    print("created %d circuits" % len(cb_circ_all[0]))

    # print(cb_circ_all[0][0])
    job_set = job_manager.run(transpile(cb_circ_all[0],backend=backend,initial_layout=[1,2]), shots=shots_intc, memory=True, backend=backend, name='intc_cb', job_tags=[tag,token])
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

    cb_data["job_set_id"] = job_set_id

    data['intc_cb'][tag] = cb_data
    # test: data saving

    print(cb_data)


####################### save data
with open('data/' + filename, 'wb') as outfile:
    pickle.dump(data, outfile)
