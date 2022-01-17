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
import CB_process
from statistics import stdev
import itertools
from qiskit.compiler import transpile

'''interleaved CB or std CB'''
use_std_CB = False

C_max = 0
shots_max = 0
use_boostrap = False

#Change test

use_density_matrix = False # density matrix based / measurement based simulation




#### Set one and only one of the following to be true:
use_ibmq = False # run on ibm devices
use_stabilizer_simulator = False # whether stabilizer simulator is used (valid only for Pauli noise)
use_density_matrix_sample = True # use density matrix simulation, but returns samples
use_state_vector_sample = False # use state vector simulation, which returns samples

#### Where is the noise channel
noise_after_the_gate = True


if use_ibmq:
    filename = 'ibmq_cb_single.txt'
else:
    filename = 'simulate_cb_single.txt'


use_readout_error = True

# True: efficient CB (simultaneously tackle with a stabilizer group), but cannot deal with degeneracy.
# stabilizer_group_cb = True

intercept_cb = True


n = 2
n_total = n
periodic = True
eps = 0.025
# eps_amp = 0.01
eps_readout = 0.005
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




if use_std_CB is True:
    unlearnable_pauli_parity_sample_list = [('ZZ',0),('ZI',1),('XX',0),('IX',1),('YZ',0),('YI',1),('XY',0),('IY',1),('ZY',0),('YX',1),('ZX',0),('YY',1)]
    learnable_pauli_parity_sample_list = [('IZ',0),('XI',0),('XZ',0)]
else:
    unlearnable_pauli_parity_sample_list = [('ZZ',0),('YI',1),('XX',0),('IY',1),('YZ',0),('ZI',1),('XY',0),('IX',1)]
    learnable_pauli_parity_sample_list = [('ZY',0),('YX',0),('ZX',0),('YY',0),('IZ',0),('XI',0),('XZ',0)]

N_un = len(unlearnable_pauli_parity_sample_list)

pauli_parity_sample_list = unlearnable_pauli_parity_sample_list + learnable_pauli_parity_sample_list



fidelity_list = {} 
stdev_list = {}
if intercept_cb is True:
    intercept_list = {}
    intercept_std_list = {}


raw_fidelity_list_all = []

for pauli_sample,parity in pauli_parity_sample_list:

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

    
    
    cb_result = CB_process.process_CB(n, C, shots, batch, Lrange, cb_data, pauli_sample = pauli_sample,repeat=repeat, periodic=periodic,use_density_matrix=use_density_matrix,intercept_cb=intercept_cb,C_max=C_max,shots_max=shots_max,use_boostrap=use_boostrap)
    raw_fidelity_list = cb_result["fidelity_list"]

    raw_fidelity_list_all.append(raw_fidelity_list)

    # print(fidelity_list)
    # print('average fidelity = %f' % np.average(fidelity_list))

    # print(raw_fidelity_list)


    # with open(filename, 'a') as f:
    #     f.write(str(raw_fidelity_list)+'\n')
    
    # we can merge raw_fidelity_list over here!

    # if use_ibmq:
    #     with open('ibmq_cb.txt', 'a') as f:
    #         f.write(str(cb_data))
    # else:
    #     with open('test_cb.txt','a') as f:
    #         f.write(str(cb_data))

    for sub_label in raw_fidelity_list.keys():
        if (sub_label,parity) in fidelity_list:
            continue
        elif(sub_label == 'I'*n):
            fidelity_list[(sub_label,parity)] = 1.0
            stdev_list[(sub_label,parity)] = 0.0
            intercept_list[(sub_label,parity)] = 1.0
            intercept_std_list[(sub_label,parity)] = 0.0

        else:
            alpha, alpha_err, intercept, intercept_err = CB_process.fit_CB_2(Lrange, raw_fidelity_list[sub_label])
            # print("CB for %dth Pauli:" % k, alpha, alpha_err)
            fidelity_list[(sub_label,parity)] = alpha
            stdev_list[(sub_label,parity)] = alpha_err
            intercept_list[(sub_label,parity)] = intercept
            intercept_std_list[(sub_label,parity)] = intercept_err

# sys.exit(0)


with open(filename, 'a') as f:
    json.dump(raw_fidelity_list_all,f)
    f.write('\n')

# Next: intepret the result from intercept_list, fidelity_list


dec_fidelity_list = {}
for k in range(0,N_un,2):
    pp1 = pauli_parity_sample_list[k]
    pp2 = pauli_parity_sample_list[k+1]
    f = (fidelity_list[pp1]+fidelity_list[pp2])/2
    r = (intercept_list[pp2]/intercept_list[pp1])
    if noise_after_the_gate is True:
        lambda1 = f * r
        lambda2 = f / r
    else:
        lambda1 = f / r
        lambda2 = f * r

    dec_fidelity_list[pp1[0]] = lambda1
    dec_fidelity_list[pp2[0]] = lambda2

for k in range(N_un,15):
    pauli_label = pauli_parity_sample_list[k]
    dec_fidelity_list[pauli_label[0]] = fidelity_list[pauli_label]
# print(dec_fidelity_list)

dec_fidelity_list['II'] = 1.0


full_pauli_list = [''.join(s) for s in itertools.product(['I','X','Y','Z'], repeat = n)]


# sys.exit(0)

# print("\nFinal result:")
# print('n =', n, ' K =', K, ' C =', C, ' Lrange =', list(Lrange))
# print("Parameters: n = %d, C = %d, " % (n,C), "L = ", str(Lrange))
if use_boostrap:
    print("Re-sampling on!")

if(C_max == 0):
    print("Parameters: n = %d, C = %d, " % (n,C))
else:
    print("Parameters: n = %d, C used = %d (tot = %d), " % (n,C_max,C))

if use_density_matrix:
    print("Density matrix based simulation")
else:
    if(shots_max==0):
        print("Measurement based simulation, shots = %d" % shots)
    else:
        print("Measurement based simulation, shots used = %d (tot = %d)" % (shots_max,shots))

if use_readout_error is True:
    print("Measurement bitflip rate = %f" % eps_readout)
else:
    print("No readout error")



print("Label / Pauli infidelity / Std(fidelity) / Intercept / Std(intercept)")
for pauli_label in pauli_parity_sample_list:
    # print(pauli_label, 1-fidelity_list[pauli_label], stdev_list[pauli_label], intercept_list[pauli_label], intercept_std_list[pauli_label])
    print(str(pauli_label)+" %.5f %.5f %.5f %.5f"%(1-fidelity_list[pauli_label], stdev_list[pauli_label], intercept_list[pauli_label], intercept_std_list[pauli_label]))
#print('Effective noise rate = ' + str(1-np.average(list(fidelity_list.values()))))



# with open(filename, 'a') as f:
#     # f.write(str(raw_fidelity_list)+'\n')
#     # f.write("Parameters: n = %d, C = %d, " % (n,C) + "L = " + str(Lrange) + '\n')
#     f.write("Parameters: n = %d, C = %d, " % (n,C) + '\n')    
#     f.write("Label / Pauli infidelity / Std(fidelity) / Intercept / Std(intercept)\n")
#     for pauli_label in pauli_parity_sample_list:
#         f.write(str(pauli_label)+', '+str(parity)+','+str(1-fidelity_list[pauli_label]) + ', ' +str(stdev_list[pauli_label]) + ', ' + str(intercept_list[pauli_label]) +', ' + str(intercept_std_list[pauli_label])+'\n')
#     #f.write('Effective noise rate = ' + str(1-np.average(list(fidelity_list.values())))+'\n')
#     f.write("-------------\n")

print("Label / Pauli infidelity (decoupled)")
for pauli_label in pauli_parity_sample_list:
# for pauli_label in full_pauli_list:
    # print(pauli_label, 1-fidelity_list[pauli_label], stdev_list[pauli_label], intercept_list[pauli_label], intercept_std_list[pauli_label])
    print(str(pauli_label[0])+" %.5f"%(1-dec_fidelity_list[pauli_label[0]]))
    # print(str(pauli_label)+" %.5f"%(1-dec_fidelity_list[pauli_label]))

