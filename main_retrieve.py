import numpy as np
import sys, json, copy, time, pickle
import qiskit
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq.managed import IBMQJobManager

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research', group='berkeley-6', project='main')
backend = provider.get_backend('ibmq_manila')
# provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')

filename = 'data/' + 'ibmq_experiment_all_20220127_0428006476'
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

with open(filename + '_full', 'wb') as outfile:
    pickle.dump(data, outfile)