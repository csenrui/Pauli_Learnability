import itertools
import numpy as np
import sys, json, copy, pickle, random
import matplotlib.pyplot as plt
import qiskit
from qiskit.result import Result
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.extensions import UnitaryGate
# from qiskit.quantum_info.operators.symplectic import pauli
from scipy.stats import sem, entropy, linregress
from scipy.optimize import curve_fit
from qiskit.quantum_info import Pauli, Clifford
# IBMQ.save_account('b3460dbc07ed93247ba3dd87b6619d71872d5d079f3f01bd5944678aa544b97203807ffcff040ca6d440ad990d907bbe59489179c190bd7b6670bf432e874940')

# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
# backend = provider.get_backend('ibmq_athens')
# backend = StatevectorSimulator()

# Lrange = list(range(1,16))

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
            # print(sub_label)
            # print(pauli_request_set)
            # print(sub_label in pauli_request_set)
            if (pauli_request_set != None) and ((sub_label in pauli_request_set) is False):
                continue
            #sub_label = sub_label[::-1]
            fidelity_list[sub_label] = {}
            for L in Lrange:
                fidelity_list[sub_label][L] = []
    else:
        fidelity_list[pauli_sample] = {}
        for L in Lrange:
            fidelity_list[pauli_sample][L] = []

    # print(pauli_sample)
    # print(fidelity_list)


    for b in range(batch):
        print(b)
        data_batch = cb_data["batch_%d" % b]
        result_batch = Result.from_dict(cb_data["result"][b])


        circuit_count = 0  ### To extract data from result_batch

        if use_boostrap:
            c_sample = []

        for i in range(len(data_batch)):
            # print("batch", b, "circuit", i)
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


            #print("n=%d" % n, "batch", b, "circuit", i)
            # print(job_data["circuit"].item().shape)
            # sys.exit(0)

            # pauliOp = Pauli(''.join(pauli_sample[::-1]))


            ###
            # circuit = construct_circuit(job_data["n"], job_data["L"], job_data["circuit"], periodic=periodic)
            # job = execute(circuit, backend, backend_options={"max_parallel_threads": 0})
            # job = execute(circuit, backend, max_parallel_threads=0)
            # state_vector = np.array(job.result().get_statevector())
            




            



            if use_density_matrix is True:
                pauliOp = Pauli(job_data["pauli"])
                rho = result_batch.data(circuit_count)['density_matrix']
                F = np.real(np.trace(rho @ pauliOp.to_matrix()))
                # print(F)
            else:
                clifford = Clifford.from_dict(job_data["clifford"])


                # assert np.mod(pauliOp.phase,2) == 0
                # phase = (-1)**(pauliOp.phase>>1)
                # # if phase == 1:
                # #     label = pauliOp.to_label()
                # # else:
                # #     label = pauliOp.to_label()[1:]
                # # assert len(label) == n
                # label = pauliOp.to_label()[-n:]

                # # make sure we deal with full weight Pauli
                # # true if m/m0 is an integer
                # for p in label:
                #     assert p!='I'



                # memory_list = result_batch.get_memory()
                # if shots_max == 0:
                #     memory = memory_list[circuit_count]
                # elif use_boostrap is False:
                #     memory = memory_list[circuit_count][:shots_max]
                # else:
                #     memory = random.sample(memory_list[circuit_count],shots_max)


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


                # print("debug")
                # print(outcomes)

                #print(outcomes)

                for sub_label in fidelity_list.keys():
                    sub_pauliOp = Pauli(sub_label)
                    sub_pauliOp = sub_pauliOp.evolve(clifford.adjoint())
                    #sub_pauliOp = sub_pauliOp.evolve(clifford)
                    assert np.mod(sub_pauliOp.phase,2) == 0
                    phase = (-1)**(sub_pauliOp.phase>>1) ### Should be here, or?
                    sub_label_evolved = sub_pauliOp.to_label()[-n:]

                    F = 0
                    tot = 0
                    for key, counts in outcomes.items(): # Walsh-Hadamard ?
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
        # "sample_to_prob_list":		sample_to_prob_list,
        # "cross_entropy_list":		cross_entropy_list,
        # "log_xeb_list":				log_xeb_list,
        # "hog_list":					hog_list,
        # "unbiased_xeb_list":		unbiased_xeb_list,
        # "unbiased_xeb_list_2":		unbiased_xeb_list_2,
        # "variance_theory_list":     variance_theory_list,
        # "variance_exp_list":        variance_exp_list,
        "fidelity_list":                     fidelity_list,
    }
    return CB_result

def rcs_fit_fun(x, a, alpha):
        return a * (alpha ** x)

def fit_CB_2(X, xeb_list):
    Y = [np.mean(xeb_list[L]) for L in X]
    Yerr = [sem(xeb_list[L]) for L in X]
    #print(linregress(X,np.log(Y)))
    params, pcov = curve_fit(rcs_fit_fun, X, Y, sigma=Yerr, absolute_sigma=True, p0=[1,1])
    #params, pcov = curve_fit(rcs_fit_fun, X, Y, absolute_sigma=True, p0=[1,1])

    alpha = params[1]
    params_err = np.sqrt(np.diag(pcov))
    alpha_err = params_err[1]

    intercept = params[0]
    intercept_err = params_err[0]
    # print(params)

    return alpha, alpha_err, intercept, intercept_err

def fit_CB(X, xeb_list):
    Y = [np.mean(xeb_list[L]) for L in X]
    Yerr = [sem(xeb_list[L]) for L in X]
    #print(linregress(X,np.log(Y)))
    try:
        params, pcov = curve_fit(rcs_fit_fun, X, Y, sigma=Yerr, absolute_sigma=True, p0=[1,1])
        alpha = params[1]
        params_err = np.sqrt(np.diag(pcov))
        alpha_err = params_err[1]

    except RuntimeError:
        alpha = 1.0
        alpha_err = 0.0

    # print(params)

    return alpha, alpha_err

    print(alpha, alpha_err)

    # params, pcov = curve_fit(rcs_fit_fun, X, Y, sigma=Ystd, absolute_sigma=False, p0=[1,1,0])

    # alpha = params[1]
    # params_err = np.sqrt(np.diag(pcov))
    # alpha_err = params_err[1] / alpha

    # print(params)

    # print(alpha, alpha_err)



    # sys.exit(0)

    # for L in range(5,14):
    # 	print(len(cross_entropy_list[L]))
    # sys.exit(0)
    
    # for L in X:
    # 	print(np.std(cross_entropy_list[L]), sem(cross_entropy_list[L]))
    # sys.exit(0)

    # print(np.split())
    

    # Y = [np.mean(entropy_list[L]) for L in X]
    # Yerr = [sem(entropy_list[L]) for L in X]

    # Y = [np.median([np.mean(S) for S in np.split(np.array(cross_entropy_list[L]),4)]) for L in X]

    
    # print(linregress(X,np.log(Y)))
    # print("slope:", (np.log(Y[14])-np.log(Y[4]))/(X[14]-X[4]))

##############################
    # X = Lrange
    # Y = [np.mean(cross_entropy_list[L]) for L in X]
    # Yerr = [sem(cross_entropy_list[L]) for L in X]
    # print(linregress(X[4:],np.log(Y[4:])))
    # plt.errorbar(X,Y,yerr=Yerr)
    # plt.plot(X, Y, label='n=%d' % n)
    # plt.xlim(X[0]-0.1,X[-1]+0.1)
    # plt.yscale("log")
    # plt.ylabel("cross entropy")
    # plt.xlabel("depth")
    # plt.legend()
    # plt.show()
################################

# job_save["n"] = n
# job_save["L"] = L
# job_save["c"] = c
# job_save["type"] = "cross_entropy"
# job_save["circuit"] = gates
# job_save["job_id"] = job_id

# data_save.append(job_save)
# num_jobs += 1
# print(num_jobs)

# data = {}
# data["data"] = data_save
# with open('data_cross_entropy.txt', 'w') as outfile:
#     json.dump(data, outfile)


# print(gates)
# print(len(gates))
# sys.exit(0)

# circuit.decompose().decompose().draw(output="mpl",filename="circuit.png")

# print(circuit.depth())

# sys.exit(0)


# job = execute(circuit, backend, shots=8192)
# job_id = job.job_id()
# result = job.result()
# # vec = result.get_statevector()

# print(result)
# sys.exit(0)

# def predict_fidelity(n,L,q,epgs_1Q,epgs_2Q,error_type="u3"):
#     F = 1
#     for i in range(L):
#         if i % 2 == 0:
#             # start from qubit 0
#             ngates = int(n/2)
#             for j in range(ngates):
#                 F *= 1 - epgs_1Q[q[2*j]][error_type] * 1.5
#                 F *= 1 - epgs_1Q[q[2*j+1]][error_type] * 1.5
#                 F *= 1 - epgs_2Q[(q[2*j],q[2*j+1])] * 1.25
#             if n%2 == 1:
#                 F *= 1 - epgs_1Q[q[n-1]][error_type] * 1.5
#                 # F *= 1 - epgs_1Q[q[n-1]]["u2"] * 1.5
#                 F *= 1 - epgs_2Q[q[n-1]][error_type] * 1.5
#         else:
#             # start from qubit 1
#             ngates = int((n-1)/2)
#             for j in range(ngates):
#                 F *= 1 - epgs_1Q[q[2*j+1]][error_type] * 1.5
#                 F *= 1 - epgs_1Q[q[2*j+2]][error_type] * 1.5
#                 F *= 1 - epgs_2Q[(q[2*j+1],q[2*j+2])] * 1.25
            
#             F *= 1 - epgs_1Q[q[0]][error_type] * 1.5
#             # F *= 1 - epgs_1Q[q[0]]["u2"] * 1.5
#             F *= 1 - epgs_2Q[q[0]][error_type] * 1.5
#             if n%2 == 0:
#                 F *= 1 - epgs_1Q[q[n-1]][error_type] * 1.5
#                 # F *= 1 - epgs_1Q[q[n-1]]["u2"] * 1.5
#                 F *= 1 - epgs_2Q[q[n-1]][error_type] * 1.5
#     return F
