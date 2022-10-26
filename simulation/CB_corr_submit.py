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

#Miss operations on K
def submit_cb(n,n_total,Lrange,C,batch,pauliList,qubit_map,gset="Pauli",repeat=None,periodic=False,use_density_matrix=False,use_reset_error=False):
	data_save = {}

	q = qubit_map
	cb_circ_all = []

	reset_id = qi.Operator([[1,0],[0,1]])

	# Correction gates
	tmp_circ = QuantumCircuit(1)
	tmp_circ.s(0)
	C0odd  = Clifford.from_circuit(tmp_circ)
	tmp_circ = QuantumCircuit(1)
	tmp_circ.sx(0)
	C1odd  = Clifford.from_circuit(tmp_circ)
	tmp_circ = QuantumCircuit(1)
	tmp_circ.z(0)
	tmp_circ.s(0)
	C0even = Clifford.from_circuit(tmp_circ)
	tmp_circ = QuantumCircuit(1)
	tmp_circ.s(0)
	tmp_circ.h(0)
	tmp_circ.s(0)
	C1even = Clifford.from_circuit(tmp_circ)
	Cl = [[C0even,C1even],[C0odd,C1odd]]



	for b in range(batch):
		# data_batch = {}
		# num_jobs = 0

		circuits_batch = []
		data_batch = []

		for l in range(len(Lrange)):
			L = Lrange[l]

			for c in range(C):
				# run the circuit
				job_save = {}
				state = QuantumCircuit(n_total,n)
				gates = QuantumCircuit(n_total,n)  ##### Any difference between n and n total?
				# circuit.reset([i for i in range(n)])
				# state preparation
				if use_reset_error:
					for j in range(n):
						state.unitary(reset_id,q[j],label='reset_id')
				for j in range(n):
					prepare_pauli_eigenstate_1q(state,q[j],pauli=pauliList[n-1-j])
					#apply_1q(circuit,q[j],gate = np.expm(pauli_sample[j]))
				state.barrier()

				for i in range(L):
					pauliLayer1 = [random.choice(['I','X','Y','Z']) for j in range(n)]
					# pauliLayer2 = [random.choice(['I','X','Y','Z']) for j in range(n)]
					#pauliTrans = Pauli(''.join(pauliLayer[::-1]))
					for j in range(n):
						# gates.pauli(pauliLayer[j],[q[j]])

						# Calculate randomized Clifford
						# cliff = Clifford.from_label(pauliLayer1[j]).dot(Cl[1][j%2])
						# gates = gates.compose(cliff.to_circuit(),[q[j]])
						
						pauli_gate_1q(gates,q[j],pauli=pauliLayer1[j])
						if j%2 == 0:
							gates.s(q[j])
						else:
							gates.sx(q[j])


					ngates = int(n/2)
					for j in range(ngates):
						gates.cx(q[2*j],q[2*j+1])
					if n%2 == 1:
						gates.id(q[n-1])

					gates.barrier()
					#pauliOp = pauliOp.evolve(pauliTrans).evolve()		


				# final layer:
				pauliLayer = [random.choice(['I','X','Y','Z']) for j in range(n)]
				# pauliTrans = Pauli(''.join(pauliLayer[::-1]))
				for j in range(n):
					#gates.pauli(pauliLayer[j],[q[j]])
					pauli_gate_1q(gates,q[j],pauli=pauliLayer[j])


				# calculate C(P), which decides our measurement setting
				# pauliOp = Pauli(''.join(pauliList[::-1])) # join in reverse order
				pauliOp = Pauli(''.join(pauliList))
				pauliOp = pauliOp.evolve(Clifford(gates).adjoint()) # note: adjoint is necessary for Heisenberg evolution.


				circuit = state.compose(gates,range(n))
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
					#print(measurement_setting)
					for j in range(n):
						measure_pauli_1q(circuit,q[j],pauli=measurement_setting[j])

					circuit.barrier()
					circuit.measure([q[i] for i in range(n)], [i for i in range(n)])
				
				#circuit.draw(filename="CB_circuit")

				### use one of the following lines:
				# circuit = circuit.decompose().decompose()
				# circuit = qiskit.transpile(circuit,optimization_level=1,basis_gates=basis_gates)
				
				# circuit.draw(output='mpl',filename='circuit3.png')
				# sys.exit(0)

				R = 1
				if repeat is not None:
					R = repeat[l]
				for r in range(R):
					circuits_batch.append(circuit)

				job_save["n"] = n
				job_save["L"] = L
				job_save["c"] = c
				#job_save["type"] = "cross_entropy_H"
				job_save["circuit"] = circuit.qasm
				job_save["clifford"] = Clifford(gates).to_dict()
				job_save["pauli"] = pauliOp.to_label()
				job_save["repeat"] = R

				# job_save["job_id"] = job_id

				data_batch.append(job_save)
				# num_jobs += 1
				# print(num_jobs)

		cb_circ_all.append(circuits_batch)
		data_save["batch_%d" % b] = data_batch
	return data_save, cb_circ_all

