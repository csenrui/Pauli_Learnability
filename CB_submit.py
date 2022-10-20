import numpy as np
import sys, json, pickle, random
import qiskit
from qiskit import IBMQ, QuantumCircuit, execute
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Pauli, Clifford
from scipy.stats import sem, unitary_group
from scipy.linalg import sqrtm,expm
import qiskit.quantum_info as qi

# IBMQ.save_account('...')
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
# backend = provider.get_backend('ibmq_montreal')
# print(backend.job_limit())
# print(backend.properties())


def prepare_pauli_eigenstate_1q(circuit,index,pauli=None):
	if pauli == 'I':
		pauli = random.choice(['X','Y','Z']) 
	# note: For Pauli I, prepare an arbitrary eigenstate.
	if pauli == 'Z':
		circuit.id([index])
	elif pauli == 'X':
		circuit.h([index])
	elif pauli == 'Y':
		circuit.h([index])
		circuit.s([index])
	else:
		assert 1==0

def pauli_gate_1q(circuit,index,pauli=None): #For stabilizer simulator to work, cannot use Pauli class
	if pauli == 'I':
		circuit.id([index])
	elif pauli == 'Z':
		circuit.z([index])
	elif pauli == 'X':
		circuit.x([index])
	elif pauli == 'Y':
		circuit.y([index])
	else:
		assert 1==0

def measure_pauli_1q(circuit,index,pauli=None):
	if pauli == 'I' or pauli == 'Z':
		circuit.id([index])
	elif pauli == 'X':
		circuit.h([index])
	elif pauli == 'Y':
		circuit.s([index])
		circuit.s([index])
		circuit.s([index])
		circuit.h([index])
	else:
		assert 1==0


def submit_cb(n,n_total,Lrange,C,batch,pauliList,qubit_map,gset="Pauli",repeat=None,periodic=False,use_density_matrix=False,use_reset_error=False, delay_time = 0):
	data_save = {}
	q = qubit_map
	cb_circ_all = []
	reset_id = qi.Operator([[1,0],[0,1]])

	for b in range(batch):

		circuits_batch = []
		data_batch = []

		for l in range(len(Lrange)):
			L = Lrange[l]
			for c in range(C):
				# run the circuit
				job_save = {}
				state = QuantumCircuit(n_total,n)
				gates = QuantumCircuit(n_total,n) 
				gates_scheduling = QuantumCircuit(n_total,n)

				# state preparation
				if use_reset_error:
					for j in range(n):
						state.unitary(reset_id,q[j],label='reset_id')
				for j in range(n):
					prepare_pauli_eigenstate_1q(state,q[j],pauli=pauliList[n-1-j])
				state.barrier()


				for i in range(L):
					pauliLayer = [random.choice(['I','X','Y','Z']) for j in range(n)]
					for j in range(n):
						pauli_gate_1q(gates,q[j],pauli=pauliLayer[j])
						pauli_gate_1q(gates_scheduling,q[j],pauli=pauliLayer[j])

					if delay_time != 0:
						should_delay = True	
						for pp in pauliLayer:
							if pp == 'X' or pp == 'Y':
								should_delay = False
								break
						if should_delay is True:
							gates_scheduling.delay(delay_time,unit='s')

					ngates = int(n/2)
					for j in range(ngates):
						gates.cx(q[2*j],q[2*j+1])
						gates_scheduling.cx(q[2*j],q[2*j+1])
					if n%2 == 1:
						gates.id(q[n-1])
						gates_scheduling.id(q[n-1])

					gates.barrier()
					gates_scheduling.barrier()

				# final layer:
				pauliLayer = [random.choice(['I','X','Y','Z']) for j in range(n)]
				for j in range(n):
					pauli_gate_1q(gates,q[j],pauli=pauliLayer[j])
					pauli_gate_1q(gates_scheduling,q[j],pauli=pauliLayer[j])


				# calculate C(P), which decides our measurement setting
				pauliOp = Pauli(''.join(pauliList))
				pauliOp = pauliOp.evolve(Clifford(gates).adjoint()) # note: adjoint is necessary for Heisenberg evolution.


				circuit = state.compose(gates_scheduling,range(n))
				if use_density_matrix:
					circuit.save_density_matrix()
				else:
					circuit.barrier()

					# measurement for fidelity estimation
					measurement_setting = pauliOp.to_label()
					# there could be a '-' in the Pauli label
					if measurement_setting[0].isupper() is False:
						measurement_setting = measurement_setting[1:]
					measurement_setting = measurement_setting[::-1]
					for j in range(n):
						measure_pauli_1q(circuit,q[j],pauli=measurement_setting[j])

					circuit.barrier()
					circuit.measure([q[i] for i in range(n)], [i for i in range(n)])
			

				R = 1
				if repeat is not None:
					R = repeat[l]
				for r in range(R):
					circuits_batch.append(circuit)

				job_save["n"] = n
				job_save["L"] = L
				job_save["c"] = c
				job_save["circuit"] = circuit.qasm
				job_save["clifford"] = Clifford(gates).to_dict()
				job_save["pauli"] = pauliOp.to_label()
				job_save["repeat"] = R

				data_batch.append(job_save)


		cb_circ_all.append(circuits_batch)
		data_save["batch_%d" % b] = data_batch
	return data_save, cb_circ_all

