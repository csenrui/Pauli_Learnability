import itertools
import numpy as np
import sys, json, copy, pickle, random
import matplotlib.pyplot as plt
import qiskit
from qiskit.result import Result
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.extensions import UnitaryGate
from scipy.stats import sem, entropy, linregress
from scipy.optimize import curve_fit
from qiskit.quantum_info import Pauli, Clifford
# IBMQ.save_account('...')
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
# backend = StatevectorSimulator()

def process_CB(n, C, shots, batch, Lrange, cb_data, pauli_sample, pauli_request_set = None, counts_batch=None, repeat=None, periodic=False, use_density_matrix = False,intercept_cb = False, C_max = 0, shots_max = 0, use_boostrap = False):


    fidelity_list = {}   
    if intercept_cb is False:
        sub_pauli_index_list = itertools.product([0,1],repeat=n)
        for index in sub_pauli_index_list:
            sub_label = str()
            for j in range(n):
                if(index[j]):
                    sub_label += pauli_sample[j]
                else:
                    sub_label += 'I'

            if (pauli_request_set != None) and ((sub_label in pauli_request_set) is False):
                continue

            fidelity_list[sub_label] = {}
            for L in Lrange:
                fidelity_list[sub_label][L] = []
    else:
        fidelity_list[pauli_sample] = {}
        for L in Lrange:
            fidelity_list[pauli_sample][L] = []



    for b in range(batch):
        data_batch = cb_data["batch_%d" % b]
        result_batch = (cb_data["result"][b])


        circuit_count = 0  ### To extract data from result_batch

        if use_boostrap:
            c_sample = []

        for i in range(len(data_batch)):
            job_data = data_batch[i]
            n = job_data["n"]
            L = job_data["L"]
            c = job_data["c"]


            if repeat is None:
                R = 1
            else:
                R = job_data["repeat"]


            if (C_max != 0):
                if (use_boostrap is False):
                    if c>=C_max:
                        circuit_count += R # unused circuits
                        continue
                else:
                    if c==0:
                        c_sample = random.sample(range(C),C_max)
                    if c not in c_sample:
                        circuit_count += R # unused circuits
                        continue


            if use_density_matrix is True:
                pauliOp = Pauli(job_data["pauli"])
                rho = result_batch.data(circuit_count)['density_matrix']
                F = np.real(np.trace(rho @ pauliOp.to_matrix()))
            else:
                clifford = Clifford.from_dict(job_data["clifford"])

                if shots_max == 0:
                    memory = result_batch.get_memory(circuit_count)
                elif use_boostrap is False:
                    memory = result_batch.get_memory(circuit_count)[:shots_max]
                else:
                    memory = random.sample(result_batch.get_memory(circuit_count),shots_max)

                outcomes = {}
                for key in memory:
                    if key in outcomes:
                        outcomes[key] += 1
                    else:
                        outcomes[key] = 1


                for sub_label in fidelity_list.keys():
                    sub_pauliOp = Pauli(sub_label)
                    sub_pauliOp = sub_pauliOp.evolve(clifford.adjoint())
                    assert np.mod(sub_pauliOp.phase,2) == 0
                    phase = (-1)**(sub_pauliOp.phase>>1) 
                    sub_label_evolved = sub_pauliOp.to_label()[-n:]

                    F = 0
                    tot = 0
                    for key, counts in outcomes.items(): # Walsh-Hadamard
                        F_key = 1
                        for j in range(n):
                            if sub_label_evolved[j]!='I' and key[j] == '1':
                                F_key *= -1
                        F += F_key * counts
                        tot += counts

                    # compute phase correction:
                    F = F*phase/tot
                
                    fidelity_list[sub_label][L].append(F)

                


            circuit_count += R



    CB_result = {
        "fidelity_list":                     fidelity_list,
    }
    return CB_result

def rcs_fit_fun(x, a, alpha):
        return a * (alpha ** x)

def fit_CB(X, xeb_list):
    Y = [np.mean(xeb_list[L]) for L in X]
    Yerr = [sem(xeb_list[L]) for L in X]

    params, pcov = curve_fit(rcs_fit_fun, X, Y, sigma=Yerr, absolute_sigma=True, p0=[1,1])


    alpha = params[1]
    params_err = np.sqrt(np.diag(pcov))
    alpha_err = params_err[1]

    return alpha, alpha_err

def fit_CB_all(X, xeb_list):
    Y = [np.mean(xeb_list[L]) for L in X]
    Yerr = [sem(xeb_list[L]) for L in X]

    params, pcov = curve_fit(rcs_fit_fun, X, Y, sigma=Yerr, absolute_sigma=True, p0=[1,1])

    return params, pcov

def calculate_uncertainty(ave1,ave2,cov1,cov2):
    int1,lam1 = ave1
    int2,lam2 = ave2
    alam = (lam1 + lam2) / 2
    vint1,vlam1 = np.diag(cov1)
    vint2,vlam2 = np.diag(cov2)
    covar1 = cov1[0,1]
    covar2 = cov2[0,1]
    # Formula by Taylor expansion
    v1 = int2**2/int1**2/4 * (vlam1 + vlam2) + alam**2*int2**2/int1**4 * vint1 + alam**2/int1**2 * vint2 + \
        alam * (-int2**2/int1**3 * covar1 + int2/int1**2 * covar2)
    v2 = int1**2/int2**2/4 * (vlam1 + vlam2) + alam**2*int1**2/int2**4 * vint2 + alam**2/int2**2 * vint1 + \
        alam * (-int1**2/int2**3 * covar2 + int1/int2**2 * covar1)
    return np.sqrt(v1), np.sqrt(v2)

