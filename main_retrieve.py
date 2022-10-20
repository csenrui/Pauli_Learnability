import numpy as np
import sys, json, copy, time, pickle
import qiskit
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq.managed import IBMQJobManager

IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-research', group='berkeley-6', project='main')
# provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='sys-reserve')
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='phy164')
# backend = provider.get_backend('ibmq_manila')
backend = provider.get_backend('ibm_perth')



# # filename_label = 'ibmq_experiment_all_20220228_7658906293' #exp2
# filename_label = 'ibmq_experiment_all_20220228_8530634712' #exp1

filename = 'data/' + 'ibmq_experiment_all_20220228_8530634712' #exp1
with open(filename, 'rb') as outfile:
    data = pickle.load(outfile)
token = data["token"]

job_manager = IBMQJobManager()

cb = data['cb']
for tag in cb:
    print(tag)
    job_set = job_manager.retrieve_job_set(job_set_id=cb[tag]['job_set_id'], provider=provider,refresh=True)
    results = job_set.results().combine_results()
    print(results)
    data['cb'][tag]["result"] = [results]

int_cb = data['int_cb']
for tag in int_cb:
    print(tag)
    job_set = job_manager.retrieve_job_set(job_set_id=int_cb[tag]['job_set_id'], provider=provider,refresh=True)
    results = job_set.results().combine_results()
    print(results)
    data['int_cb'][tag]["result"] = [results]

intc_cb = data['intc_cb']
for tag in intc_cb:
    print(tag)
    job_set = job_manager.retrieve_job_set(job_set_id=intc_cb[tag]['job_set_id'], provider=provider,refresh=True)
    results = job_set.results().combine_results()
    print(results)
    data['intc_cb'][tag]["result"] = [results]

with open(filename + '_full', 'wb') as outfile:
    pickle.dump(data, outfile)
