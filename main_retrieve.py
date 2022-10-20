import numpy as np
import sys, json, copy, time, pickle
import qiskit
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.ibmq.managed import IBMQJobManager


# input ibmq credential
IBMQ.load_account()
provider = IBMQ.enable_account("...",hub='...', group='...', project='...')
backend = provider.get_backend('ibmq_montreal')


filename = 'data/' + 'ibmq_experiment_all_20220323_8530634712' 
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
