#!/usr/bin/env python
# coding: utf-8

# Copyright 2021, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from qiskit import transpile
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, XGate, YGate
from qiskit.circuit.library import SwapGate, PermutationGate
from qrac.qrac_parametervector import full_circuit_21p_qrac, full_circuit_31p_qrac, basis_encoding
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B, spsa
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.algorithms.regressors import  NeuralNetworkRegressor
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, InstructionProperties
from qiskit.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

from paulitwirl import PauliTwirl

from libraries.estimator_qnn import  EstimatorQNN


def basis_encoding(len_bitstring: int) -> QuantumCircuit:
    num_qubits = len_bitstring
    params = ParameterVector('a', length=len_bitstring)

    qr = QuantumRegister(num_qubits, 'Basis Encoding')
    circ = QuantumCircuit(qr)

    for i in range(num_qubits):
        param = params[i]
        theta = np.pi * param
        circ.rx(theta, qr[i])

    return circ

def transpile_circuit( circuit, opt_level, backend, initial_layout, PT = False):
    pass_manager = generate_preset_pass_manager(
        optimization_level=opt_level,
        backend=backend,
        seed_transpiler=42,
        initial_layout=initial_layout,
    )
    target = backend.target
    basis_gates = list(target.operation_names)

    t_qc = pass_manager.run(circuit)

    # Pauli Twirling
    if PT:
        pt_pm = PassManager([PauliTwirl()])
        qc_pt = pt_pm.run(t_qc)
        t_qc = transpile(qc_pt, backend=backend, optimization_level=0)
        
    # Dynamic decoupling
    X = XGate()
    Y = YGate()

    dd_sequence = [X, Y, X, Y]

    y_gate_properties = {}
    for qubit in range(target.num_qubits):
        y_gate_properties.update(
            {
                (qubit,): InstructionProperties(
                    duration=target["x"][(qubit,)].duration,
                    error=target["x"][(qubit,)].error,
                )
            }
        )

    target.add_instruction(YGate(), y_gate_properties)

    dd_pm = PassManager(
        [
            ALAPScheduleAnalysis(target=target),
            PadDynamicalDecoupling(target=target, dd_sequence=dd_sequence),
        ]
    )

    qc_dd = dd_pm.run(t_qc)
    qc_dd = BasisTranslator(sel, basis_gates)(qc_dd)

    return( qc_dd) 
    

def get_model(qnn, mode, optimizer, loss, max_iter, prior_iter, callback_graph, 
              initial_weights, primitive, learning_rate_a = None, perturbation_gamma = None ):

    # Only for SamplerQNN, one_hot=True
    optimizer_one_hot = (primitive == 'sampler')

    if mode == 'classification':
        if optimizer == 'COBYLA':
            model =  NeuralNetworkClassifier(
                qnn,
                loss=loss,
                optimizer=COBYLA(maxiter=max_iter),  # Set max iterations here
                callback=callback_graph,
                one_hot=optimizer_one_hot,
                initial_point=initial_weights,
        )
        elif optimizer == 'SPSA':
            
            if (learning_rate_a != None) & (perturbation_gamma != None):
                # set up the power series
                def learning_rate():
                    return spsa.powerseries(learning_rate_a, 0.602, 0)
                gen = learning_rate()
                learning_rates = np.array([next(gen) for _ in range(max_iter + prior_iter)])
                learning_rates = learning_rates[prior_iter:(max_iter + prior_iter)]

                def perturbation():
                    return spsa.powerseries(0.2, perturbation_gamma)
                gen = perturbation()
                perturbations = np.array([next(gen) for _ in range(max_iter + prior_iter)])
                perturbations = perturbations[prior_iter:(max_iter + prior_iter)]

                model =  NeuralNetworkClassifier(
                    qnn,
                    loss=loss,
                    optimizer=SPSA(maxiter=max_iter, 
                                learning_rate= learning_rates,
                                perturbation= perturbations),
                    callback=callback_graph,
                    one_hot=optimizer_one_hot,
                    initial_point=initial_weights,
                )
            else:
                model =  NeuralNetworkClassifier(
                    qnn,
                    loss=loss,
                    optimizer=SPSA(maxiter=max_iter),
                    callback=callback_graph,
                    one_hot=optimizer_one_hot,
                    initial_point=initial_weights,
                )
        elif optimizer == 'L_BFGS_B':
            model =  NeuralNetworkClassifier(
                qnn,
                loss=loss,
                optimizer=L_BFGS_B(maxiter=max_iter),  # Set max iterations here
                callback=callback_graph,
                one_hot=optimizer_one_hot,
                initial_point=initial_weights,
        )
        else:
            raise ValueError('Invalid optimizer.')
    elif mode == 'regression':
        if optimizer == 'COBYLA':
            model =  NeuralNetworkRegressor(
                qnn,
                loss=loss,
                optimizer=COBYLA(maxiter=max_iter),  # Set max iterations here
                callback=callback_graph,
                initial_point=initial_weights,
        )
        elif optimizer == 'SPSA':
            if (learning_rate_a != None) & (perturbation_gamma != None):
                # set up the power series
                def learning_rate():
                    return spsa.powerseries(learning_rate_a, 0.602, 0)
                gen = learning_rate()
                learning_rates = np.array([next(gen) for _ in range(max_iter + prior_iter)])
                learning_rates = learning_rates[prior_iter:(max_iter + prior_iter)]

                def perturbation():
                    return spsa.powerseries(0.2, perturbation_gamma)
                gen = perturbation()
                perturbations = np.array([next(gen) for _ in range(max_iter + prior_iter)])
                perturbations = perturbations[prior_iter:(max_iter + prior_iter)]
                
                model =  NeuralNetworkRegressor(
                    qnn,
                    loss=loss,
                    optimizer=SPSA(maxiter=max_iter, 
                                learning_rate= learning_rates,
                                perturbation= perturbations),
                    callback=callback_graph,
                    initial_point=initial_weights,
            )
        elif optimizer == 'L_BFGS_B':
            model =  NeuralNetworkRegressor(
                qnn,
                loss=loss,
                optimizer=L_BFGS_B(maxiter=max_iter),  # Set max iterations here
                callback=callback_graph,
                initial_point=initial_weights,
        )
        else:
            raise ValueError('Invalid optimizer.')  
        
    return model



def get_qnn(backend_name, backend, primitive, num_qubits, circuit, feature_map, ansatz, custom_interpret, 
            estimator = None, sampler = None, pad_observable = False, pyramid_circuit = True):

    if (primitive == 'estimator') or ('ibm' in backend_name):
        # Observable for EstimatorQNN
        if pad_observable:
            observable = SparsePauliOp.from_list([("Z" + "I" * (int(num_qubits) - 1), 0.5), ("I" * (int(num_qubits)), 0.5)])
        elif pyramid_circuit:
            center = int(num_qubits/2) - 1
            observable = SparsePauliOp.from_list([( "I" * (center) + "Z" + "I" * (num_qubits - (center+1)), 0.5)])
        else:
            observable = SparsePauliOp.from_list([("Z" + "I" * (int(num_qubits) - 1), 0.5)])
        
        if ('ibm' in backend_name):
            observable = observable.apply_layout(circuit.layout, num_qubits=backend.num_qubits)

        # we decompose the circuit for the QNN to avoid additional data copying
        qnn = EstimatorQNN(
            estimator=estimator,
            circuit=circuit,
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
    elif primitive == 'sampler':
        # Custom interpret for SamplerQNN
        def custom_interpreter(i, bit_len=num_qubits):
            if custom_interpret == 'last_qubit':
                # Dynamically format based on actual bit length
                format_str = f"{{:0{bit_len}b}}"
                return int(format_str.format(i)[0])
            elif custom_interpret == 'parity':
                return "{:b}".format(i).count("1") % 2
            else:
                raise ValueError('Invalid custom interpret.')

         # we decompose the circuit for the QNN to avoid additional data copying
        qnn = SamplerQNN(
            sampler=sampler,
            circuit=circuit.decompose(),
            interpret=custom_interpreter,
            output_shape=2,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
    else:
        raise ValueError('Invalid qiskit-aer primitive or backend')

    return qnn






# Create complete QCNN circuit
def create_qcnn_circuit(feature_map, featuerMap_reps, entanglement, train_data, test_data, 
                        pool_until =1, pool_circuit = 0, conv_circuit = 0, pyramid_circuit = True, primitive = 'estimator'):
    
    # Create Feature Map
    feature_map, train_data, test_data, num_qubits =  create_feature_map(feature_map, 
                                                                               featuerMap_reps,
                                                                               entanglement, 
                                                                               train_data, 
                                                                               test_data)
        
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    
    # Create Ansatz    
    if pyramid_circuit:
        ansatz = create_pyramid_ansatz(ansatz=ansatz,
                            num_qubits=num_qubits,
                            pool_until=pool_until,
                            pool_circuit = pool_circuit,
                            conv_circuit = conv_circuit)
    else:
        ansatz = create_ansatz(ansatz=ansatz,
                            num_qubits=num_qubits,
                            pool_until=pool_until,
                            pool_circuit = pool_circuit,
                            conv_circuit = conv_circuit)

    # Combining the feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    if (primitive == 'sampler'):
        circuit.measure_all()

    return circuit, ansatz, feature_map, num_qubits, featuerMap_reps, train_data, test_data


def create_pyramid_ansatz(ansatz, num_qubits, pool_until =1, pool_circuit = 0, conv_circuit = 0, barrier = False):
    
    # number and size of convolutional layters
    x = num_qubits
    conv_layer_sizes = []
    while x > 2:
        conv_layer_sizes.append(x)
        x = int( np.ceil(np.ceil(x/2) / 2.) * 2)

    center = int(num_qubits/2)
    
    for layer in range(len(conv_layer_sizes)):
        nqbits = conv_layer_sizes[layer]
        if nqbits % 4 == 0: # TODO Principle confirm number of pooling gates
            ansatz.compose(conv_layer(num_qubits=nqbits,
                                                    param_prefix=f"c{layer}",
                                                    conv_ansatz=conv_circuit),
                            list(range(center - int(nqbits/2), center + int(nqbits/2))),
                            inplace=True)
        else:
            ansatz.compose(conv_layer_second(num_qubits=nqbits,
                                             param_prefix=f"cs{layer}"),
                           list(range(center - int(nqbits/2), center + int(nqbits/2))),
                           inplace=True)
            ansatz.compose(conv_layer_first(num_qubits=nqbits, 
                                            param_prefix=f"cf{layer}"),
                            list(range(center - int(nqbits/2), center + int(nqbits/2))),
                            inplace=True)
        if barrier:           
            ansatz.barrier()

        # Pooling Layer
        midpoint = int( np.floor(np.floor(nqbits/2) / 2.) * 2)
        endpoint = midpoint * 2
        ansatz.compose(pool_layer(sources = list(range(0, midpoint-1, 2)) + list(range(midpoint+1, endpoint, 2)),
                                  sinks =   list(range(1, midpoint, 2)) + list(range(midpoint, endpoint-1, 2)),
                                  param_prefix=f"p{layer}",
                                 pool_ansatz=pool_circuit),
                        list(range(center - int(endpoint/2), center + int(endpoint/2))),
                        inplace=True)
        if barrier:           
            ansatz.barrier()    
                
        # Swap pyramid
        swap_size = int( np.ceil((np.ceil(nqbits/2)-2) / 2.) * 2)
        if swap_size > 0:
            ansatz.compose(swap_pyramid(num_qubits=swap_size),
                                        list(range(center-swap_size-1, center - 1)),
                                        inplace=True)
            ansatz.compose(swap_pyramid(num_qubits=swap_size),
                                        list(range(center +1 , center + swap_size + 1)),
                                        inplace=True)
            if barrier:           
                ansatz.barrier()
                
    # For the last layer with 2 qubits
    layer = layer + 1
    ansatz.compose(conv_layer(num_qubits=2,
                              param_prefix=f"c{layer}"),
                   list(range(center - 1, center + 1)),
               inplace=True)
    if barrier:           
        ansatz.barrier()
        
    # Pooling Layer
    ansatz.compose(pool_layer(sources=[0], 
                            sinks=[1], 
                            param_prefix=f"p{layer}"),
                            list(range(center - 1, center + 1)),
                inplace=True)

    return ansatz

def create_ansatz(ansatz, num_qubits, pool_until =1, pool_circuit = 0, conv_circuit = 0):
    
    # ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # Convolutional layer acts on "full_qubits" number of qubits
    # Pooling layer reduces "full_qubits" number of qubits to "half_qubits" number of qubits
    # If "full_qubits" is odd, take the larger half to ensure last pooling layer has 2->1 qubits
    full_qubits = num_qubits
    half_qubits = (num_qubits + 1) // 2

    # Add convolutional and pooling layers until there is only one qubit left for binary classification
    layer = 1

    while full_qubits >pool_until:
        # Convolutional Layer
        ansatz.compose(conv_layer(num_qubits=full_qubits,
                                               param_prefix=f"c{layer}",
                                               conv_ansatz=conv_circuit),
                       list(range(num_qubits - full_qubits, num_qubits)),
                       inplace=True)

        # Pooling Layer
        ansatz.compose(pool_layer(sources=list(range(0, half_qubits)),
                                               sinks=list(range(half_qubits, full_qubits)),
                                               param_prefix=f"p{layer}",
                                               pool_ansatz=pool_circuit),
                       list(range(num_qubits - full_qubits, num_qubits)),
                       inplace=True)

        full_qubits = half_qubits
        half_qubits = (half_qubits + 1) // 2
        layer += 1

    if pool_until != 1:
        ansatz.compose(fully_connected(num_qubits=full_qubits),
                       list(range(num_qubits - full_qubits, num_qubits)),
                       inplace=True)
    
    return ansatz
    

def create_feature_map(feature_map, featuerMap_reps, entanglement, train_data, test_data):
    if feature_map == 'Z':
        reps = featuerMap_reps
        num_qubits = train_data.shape[1]
        feature_map = ZFeatureMap(num_qubits, reps=reps, parameter_prefix='a')
        # Change the data encoding from 1 to pi/2 when using Z or ZZ Feature Map rotation angles
        train_data = train_data.astype(float)
        train_data[train_data == 1] = np.pi / 2
        test_data = test_data.astype(float)
        test_data[test_data == 1] = np.pi / 2
    elif feature_map == 'ZZ':
        reps = featuerMap_reps
        num_qubits = train_data.shape[1]
        feature_map = ZZFeatureMap(num_qubits, reps=reps, entanglement=entanglement, parameter_prefix='a')
        # Change the data encoding from 1 to pi/2 when using Z or ZZ Feature Map rotation angles
        train_data = train_data.astype(float)
        train_data[train_data == 1] = np.pi / 2
        test_data = test_data.astype(float)
        test_data[test_data == 1] = np.pi / 2
    elif feature_map == 'qrac_21':
        len_bitstring = train_data.shape[1]
        assert len_bitstring % 2 == 0, 'Length of bitstring needs to be an even number for (2,1,p)-QRAC.'
        feature_map = full_circuit_21p_qrac(len_bitstring)
        num_qubits = int(len_bitstring / 2)
    elif feature_map == 'qrac_31':
        len_bitstring = train_data.shape[1]
        if len_bitstring == 8:
            pad = np.zeros((train_data.shape[0], 1))
            train_data = np.hstack((train_data, pad))
            pad = np.zeros((test_data.shape[0], 1))
            test_data = np.hstack((test_data, pad))
        assert len_bitstring % 3 == 0, 'Length of bitstring needs to be a multiple of 3 for (3,1,p)-QRAC.'
        feature_map = full_circuit_31p_qrac(len_bitstring)
        num_qubits = int(len_bitstring / 3)
    elif feature_map == 'basis':
        len_bitstring = train_data.shape[1]
        feature_map = basis_encoding(len_bitstring)
        num_qubits = len_bitstring
    else:
        raise ValueError('Feature Map not valid.')
    
    return feature_map, train_data, test_data, num_qubits



def conv_circuit(params, conv_ansatz=0):
    if conv_ansatz == 0:
        # two qubit unitary N as defined in Vatan and Williams
        assert len(params) == 3
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)

    elif conv_ansatz == 1:
        # full KAK decomposition
        assert len(params) == 15
        target = QuantumCircuit(2)
        target.rz(params[0], 0)
        target.rz(params[1], 1)
        target.ry(params[2], 0)
        target.ry(params[3], 1)
        target.rz(params[4], 0)
        target.rz(params[5], 1)
        #target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[6], 0)
        target.ry(params[7], 1)
        target.cx(0, 1)
        target.ry(params[8], 1)
        target.cx(1, 0)
        #target.rz(np.pi / 2, 0)
        target.rz(params[9], 0)
        target.rz(params[10], 1)
        target.ry(params[11], 0)
        target.ry(params[12], 1)
        target.rz(params[13], 0)
        target.rz(params[14], 1)

    else:
        raise AssertionError('Invalid Ansatz number.')

    return target


def conv_layer(num_qubits, param_prefix, conv_ansatz=0, periodic=False, test=False):
    num_params = [3, 15]  # in order corresponding to each different type of conv ansatz
    num_param = num_params[conv_ansatz]
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * num_param)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:(param_index + num_param)], conv_ansatz), [q1, q2])
        if test:
            qc.barrier()
        param_index += num_param
        
    if periodic:
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index:(param_index + num_param)], conv_ansatz), [q1, q2])
            if test:
                qc.barrier()
            param_index += num_param
    else:
        for q1, q2 in zip(qubits[1::2], qubits[2::2]):  # no periodic connectivity
            qc = qc.compose(conv_circuit(params[param_index:(param_index + num_param)], conv_ansatz), [q1, q2])
            if test:
                qc.barrier()
            param_index += num_param

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def conv_layer_first(num_qubits, param_prefix, conv_ansatz=0, periodic=False, test=False):
    num_params = [3, 15]  # in order corresponding to each different type of conv ansatz
    num_param = num_params[conv_ansatz]
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer First")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * num_param)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:(param_index + num_param)], conv_ansatz), [q1, q2])
        if test:
            qc.barrier()
        param_index += num_param

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def conv_layer_second(num_qubits, param_prefix, conv_ansatz=0, periodic=False, test=False):
    num_params = [3, 15]  # in order corresponding to each different type of conv ansatz
    num_param = num_params[conv_ansatz]
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer Second")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * num_param)
    if periodic:
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index:(param_index + num_param)], conv_ansatz), [q1, q2])
            if test:
                qc.barrier()
            param_index += num_param
    else:
        for q1, q2 in zip(qubits[1::2], qubits[2::2]):  # no periodic connectivity
            qc = qc.compose(conv_circuit(params[param_index:(param_index + num_param)], conv_ansatz), [q1, q2])
            if test:
                qc.barrier()
            param_index += num_param

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params, pool_ansatz=0):
    if pool_ansatz == 0:
        # Ansatz from the qiskit tutorial
        assert len(params) == 3
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)

    elif pool_ansatz == 1:
        # Non-parametric dynamic circuit (Tanvi's version)
        assert len(params) == 0
        target = QuantumCircuit(2, 1)
        target.cx(1, 0)
        target.h(0)
        target.measure(0, 0)  # measure qubit 0 and store it's classical bit value in cbit[0]
        with target.if_test((0, 1)):  # if measured state of cbit[0] is true, flip qubit 1
            target.x(1)

    elif pool_ansatz == 2:
        # From "Quantum convolutional neural network for classical data classification"
        assert len(params) == 2
        target = QuantumCircuit(2)
        target.crz(params[0], 0, 1)
        target.x(0)
        target.crx(params[1], 0, 1)

    elif pool_ansatz == 3:
        # Similar to controlled-U3 rotation
        assert len(params) == 3
        target = QuantumCircuit(2)
        target.rz(params[0], 1)
        target.cx(0, 1)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.rz(params[2], 1)

    else:
        raise AssertionError('Invalid Ansatz number.')

    return target


def pool_layer(sources, sinks, param_prefix, pool_ansatz=0, test=False):
    dynamic_circuit = [False, True, False, False]  # in order corresponding to each different type of pool ansatz
    dc = dynamic_circuit[pool_ansatz]
    num_params = [3, 0, 2, 3]  # in order corresponding to each different type of pool ansatz
    num_param = num_params[pool_ansatz]
    if dc:
        num_qubits = len(sources) + len(sinks)
        num_clbits = len(sources)
        qc = QuantumCircuit(num_qubits, num_clbits, name="Pooling Layer")
        params = []
        clbit_index = 0
        for source, sink in zip(sources, sinks):
            qc.compose(pool_circuit(params, pool_ansatz), qubits=[source, sink], clbits=[clbit_index], inplace=True)
            clbit_index += 1
            if test:
                qc.barrier()
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits, num_clbits)
        qc.append(qc_inst, range(num_qubits), range(num_clbits))

    else:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * num_param)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index: (param_index + num_param)], pool_ansatz), [source, sink])
            if test:
                qc.barrier()
            param_index += num_param
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))

    return qc


def swap_pyramid(num_qubits):
    assert num_qubits % 2 == 0, "Even number of qubits required"
    qc = QuantumCircuit(num_qubits, name="Swap Pyramid")
    initial_qubits = list(range(num_qubits))

    qubits = list(range(num_qubits))
    while len(qubits) > 0:
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(SwapGate(), [q1, q2])
        qubits = qubits[1:-1]

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, initial_qubits)
    return qc

def perm_pyramid(num_qubits):
    assert num_qubits % 2 == 0, "Even number of qubits required"
    qc = QuantumCircuit(num_qubits, name="Perm Pyramid")
    initial_qubits = list(range(num_qubits))

    qubits = list(range(num_qubits))
    while len(qubits) > 0:
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(PermutationGate([q2, q1]), [q1, q2])
        qubits = qubits[1:-1]

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, initial_qubits)
    return qc

def fully_connected(num_qubits, test=False):
    qc = QuantumCircuit(num_qubits)
    two_local = TwoLocal(num_qubits, ['ry', 'rz'], 'cx', 'pairwise', reps=1,
                         parameter_prefix='θ', insert_barriers=test, name='Fully Connected')
    qc.compose(two_local, inplace=True)

    return qc


def test_functions():
    params1 = ParameterVector("θ", length=3)
    circuit1 = conv_circuit(params=params1, conv_ansatz=0)
    print(circuit1)

    params2 = ParameterVector("θ", length=15)
    circuit2 = conv_circuit(params=params2, conv_ansatz=1)
    print(circuit2)

    circuit3 = conv_layer(num_qubits=4, param_prefix="θ", conv_ansatz=0, periodic=False, test=True)
    print(circuit3.decompose())

    circuit4 = conv_layer(num_qubits=4, param_prefix="θ", conv_ansatz=0, periodic=True, test=True)
    print(circuit4.decompose())

    circuit5 = conv_layer(num_qubits=4, param_prefix="θ", conv_ansatz=1, periodic=False, test=True)
    print(circuit5.decompose())

    params6 = ParameterVector("θ", length=3)
    circuit6 = pool_circuit(params=params6, pool_ansatz=0)
    print(circuit6)

    params7 = []
    circuit7 = pool_circuit(params=params7, pool_ansatz=1)
    print(circuit7)

    sources8 = [0, 1, 2, 6]
    sinks8 = [3, 4, 5, 7]
    circuit8 = pool_layer(sources=sources8, sinks=sinks8, param_prefix="θ", pool_ansatz=0, test=True)
    print(circuit8.decompose())

    sources9 = [0, 1, 2, 6]
    sinks9 = [3, 4, 5, 7]
    circuit9 = pool_layer(sources=sources9, sinks=sinks9, param_prefix="θ", pool_ansatz=3, test=True)
    print(circuit9.decompose())

    circuit10 = fully_connected(num_qubits=6, test=True)
    print(circuit10.decompose())



