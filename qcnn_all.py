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

from IPython.display import clear_output
import numpy as np
import logging
import hydra
import pickle
import os
import re

from qiskit_aer import AerSimulator
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2 as Estimator
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from scipy.stats import pearsonr
from collections import Counter

from libraries import mylib, qcnn_lib

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


@hydra.main(config_path='configs', config_name='defaults_qcnn_all.yaml', version_base='1.1')
def main(args):
    log = logging.getLogger(__name__)
    log.info(f"\nProgram started at {mylib.get_time()}")
    
    if ('ibm' in args['backend']) and (args['primitive'] == 'sampler'):
        raise ValueError('Invalid primitive with hardware. Must use estimator.')  
    if (args['mode'] == 'regression') and (args['primitive'] == 'sampler'):
        raise ValueError('Invalid primitive with regression. Must use estimator.')  
    if (args['mode'] == 'custom_interpret') and (args['primitive'] == 'estimator'):
        raise ValueError('Invalid primitive with custom_interpret. Must use sampler.')  

    seed = args['seed']
    algorithm_globals.random_seed = seed
    np.random.seed(seed)
    log.info(f"Seed: {seed}")

    mylib.root = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, os.path.normpath(args["dir_root"]))
    mylib.output = os.path.join(mylib.root, args['dir_output'])

    if not os.path.exists(mylib.output):
        os.mkdir(mylib.output)

    # ---------------------------------------------------------------------------------
    # Preprocess data
    log.info(f"Train data file: {args['file_train_data']}")
    log.info(f"Test data file: {args['file_test_data']}")
    train_data, test_data, train_labels, test_labels,  num_class, num_motifs = mylib.preprocess_data(dir_root=mylib.root, args=args)

    log.info(f"Motifs in positions: {args['motifs_to_use']}")
    log.info(f"Train labels count: {Counter(train_labels)}")
    log.info(f"Test labels count: {Counter(test_labels)}")


    # Encode the data
    log.info(f"Encoding method : {args['encoder']}")
    train_data, test_data = mylib.data_encoder(args, train_data, test_data, num_class, num_motifs)

    # ---------------------------------------------------------------------------------
    # QUANTUM

    # Create feature map, ansatz, and circuit
    circuit, ansatz, feature_map, num_qubits, reps, train_data, test_data = qcnn_lib.create_qcnn_circuit(  feature_map = args['featureMap'],
                                                                                                            featuerMap_reps = args['featureMap_reps'],
                                                                                                            entanglement=args['entanglement'],
                                                                                                            train_data =train_data,
                                                                                                            test_data= test_data,
                                                                                                            pool_until =args['pool_until'], 
                                                                                                            pool_circuit = args['pool_circuit'], 
                                                                                                            conv_circuit = args['conv_circuit'],
                                                                                                            pyramid_circuit = args['pyramid_circuit'],
                                                                                                            primitive = args['primitive']
                                                                                                        )
    log.info(f"Feature Map: {args['featureMap']}")
    log.info(f"Number of qubits: {num_qubits}")
    if args['featureMap'] == 'Z':
        log.info(f"Number of Feature Map repetitions: {reps}")
    if args['featureMap'] == 'ZZ':
        log.info(f"Number of Feature Map repetitions: {reps}")
    log.info(f"Number of Ansatz parameters: {len(ansatz.parameters)}")
    log.info(f"Conv circuit: {args['conv_circuit']}")
    log.info(f"Pool circuit: {args['pool_circuit']}")
    log.info(f"Pool until: {args['pool_until']}")
    log.info(f"Circuit depth: {circuit.depth()}")
    log.info(f"Circuit gates: {circuit.decompose().count_ops()}")
    log.info(f"Circuit gates: {np.sum(list(circuit.decompose().count_ops().values()))}")

    # QCNN
    # Specify the backend in the config file
    backend = args['backend']
    log.info(f"Backend: {args['backend']}")
    estimator = None
    sampler = None
    
    options  = {}
    options['resilience_level'] = args['resilience_level'] #For read-out error mitigation set 1
    options['optimization_level'] = args['optimization_level'] #Set the circuit optimization level
    options['default_shots'] = args['shots'] #Number of shots

    if 'device' not in args.keys():
        args['device'] = 'CPU'
        log.info(f"Qiskit-Aer primitive: {args['primitive']}")

    if backend == 'statevector':
        backend = AerSimulator(method=backend, device = args['device'])
        # session = Session(service=service, backend=backend)
        if args['primitive'] == 'estimator':
            # Estimator primitive
            estimator = StatevectorEstimator() #(options=options)
        elif args['primitive'] == 'sampler':
            # Sampler primitive
            sampler = StatevectorSampler()
        else:
            raise ValueError('Invalid qiskit-aer primitive.')
    else:
        service = QiskitRuntimeService(channel=args['channel'], instance=args['instance'])
        if 'noisy' in backend:
            backend_name = re.sub( 'noisy_', '', backend )
            real_backend = service.backend(backend_name)
            backend = AerSimulator.from_backend(real_backend, method ='statevector', device = args['device'])
            session = Session(service=service, backend=backend)
            estimator = Estimator(session=session, options=options)
            log.info(f"Noisy Backend: {backend.name}")
        elif 'ibm' in backend: #For hardware backends, must use estimator primitive
            # Read default credentials from disk
            if backend == 'ibm_least':
                backend = service.least_busy(simulator=False, operational=True, min_num_qubits=num_qubits)
            else:
                backend = service.backend(args['backend'])
            session = Session(backend=backend)
            estimator = Estimator(mode=session) #, options=options)
            log.info(f"Backend: {backend.name}")

            estimator.options.default_shots = args['shots']

            # Set simple error suppression/mitigation options
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"

            estimator.options.resilience.measure_mitigation = True    
            if args['pauliTwirling']:
                estimator.options.twirling.enable_gates = True
                estimator.options.twirling.num_randomizations = "auto"
        
    #print(ansatz.decompose())
    #print(feature_map.parameters)
    #print(feature_map.decompose())
    #print(circuit.decompose())

    # Transpile circuit if hardware backend
    if 'ibm' in args['backend']:
        if args['qbit_idx'] is None:
            # initial_layout = list(range(num_qubits))
            readoutThreshold = args['readoutThreshold']
            edgeThreshold = args['edgeThreshold']
            initial_layout, _ = mylib.get_qubit_path( backend = backend, path_length=num_qubits, 
                                                     readoutThreshold = readoutThreshold, edgeThreshold = edgeThreshold )

        else:
            initial_layout = list(args['qbit_idx'])
        log.info(f"qbit_idx: {initial_layout}")
        #circuit = qcnn_lib.transpile_circuit( circuit, 
        #                                         args['optimization_level'], 
        #                                         backend, 
        #                                         initial_layout = initial_layout, #list(range(num_qubits)), 
        #                                         PT = args['pauliTwirling'])
        pm = generate_preset_pass_manager(optimization_level=args['optimization_level'], target=backend.target, initial_layout=initial_layout, routing_method=None)
        circuit = pm.run(circuit.decompose().decompose())

        log.info(f'transpiled 2q-depth: {circuit.depth(lambda x: x.operation.num_qubits==2)}')
        log.info(f'transpiled 2q-size: {circuit.size(lambda x: x.operation.num_qubits==2)}')
            
    # Get the QNN
    pad_observable = False
    qnn = qcnn_lib.get_qnn(backend_name = args['backend'],
                               backend = backend,
                               primitive = args['primitive'],
                               num_qubits = num_qubits, 
                               circuit = circuit, 
                               feature_map = feature_map, 
                               ansatz = ansatz,
                               custom_interpret = args['custom_interpret'],
                               estimator = estimator, 
                               sampler = sampler,
                               pad_observable = pad_observable,
                               pyramid_circuit = args['pyramid_circuit'],
                               
    )


    # saves the weights with each iteration
    prior_iter = 0
    objective_func_vals = []
    def callback_graph(nfev, parameters, fvalue, stepsize, accepted):
        # Updated callback signature for qiskit-algorithms 0.4.0
        # nfev: number of function evaluations
        # parameters: current parameter values
        # fvalue: current function value
        # stepsize: current step size
        # accepted: whether the step was accepted
        weights = parameters
        obj_func_eval = fvalue
        clear_output(wait=True)
        objective_func_vals.append(obj_func_eval)
        # Capture the iteration point
        weights = np.append( weights, len(objective_func_vals) ) #+(prior_iter*2))
        np.save(os.path.join(mylib.output, args['file_weights']),weights)
        np.save(os.path.join(mylib.output, args['file_objFuncVal']),objective_func_vals)
        print(len(objective_func_vals), obj_func_eval)
        
    # load the initial weights if file exists and decrement the max iteration by the stored iteration count
    if os.path.isfile(os.path.join(mylib.output, args['file_weights'])):
        initial_weights = np.load( os.path.join(mylib.output, args['file_weights']) )
        prior_iter = int(initial_weights[-1] )
        
        if args['reset_prior_iter']:
            prior_iter = 0
        else:
            if (args['optimizer'] == 'SPSA') & (args['learning_rate_a'] != None) & (args['perturbation_gamma'] != None): # adjust when there is no calibration
                prior_iter = int(np.floor( prior_iter/ 2))
            elif (args['optimizer'] == 'SPSA'):  # control for calibration where there are 50 iterations prior to the start
                prior_iter = int(np.max([0, prior_iter - 50]))
                if prior_iter > 0:
                    prior_iter = int(np.floor(prior_iter / 2))

        args['max_iter'] = args['max_iter']-prior_iter
        if args['max_iter'] <= 0:
            args['max_iter'] = 1
        initial_weights =  initial_weights[:-1]
        print("Initial weights: ",initial_weights)
    else:
        print(f"Initial weights not found at { os.path.join(mylib.output, args['file_weights'])}!")
        prior_iter = 0
        initial_weights = None
    
    # load prior obj function valu if file exists and decrement the max iteration by the stored iteration count
    if os.path.isfile(os.path.join(mylib.output, args['file_objFuncVal'])):
        objective_func_vals = list( np.load( os.path.join(mylib.output, args['file_objFuncVal']) ) )
    else:
        print(f"Prior Objection Function Values not found at { os.path.join(mylib.output, args['file_objFuncVal'])}!")
        objective_func_vals = []

    # Get the complete NN model
    model = qcnn_lib.get_model(qnn = qnn, 
                                   mode = args['mode'],
                                   optimizer = args['optimizer'],
                                   loss = args['loss'], 
                                   max_iter = args['max_iter'], 
                                   prior_iter = prior_iter,
                                   callback_graph = callback_graph,
                                   initial_weights = initial_weights, 
                                   primitive = args['primitive'],
                                   learning_rate_a = args['learning_rate_a'],
                                   perturbation_gamma = args['perturbation_gamma'])

    log.info(f"Optimizer: {args['optimizer']}")
    log.info(f"Max iteration: {args['max_iter']}")
    log.info(f"Loss function: {args['loss']}")

    # train model
    log.info(f"Start Model Fit")
    objective_func_vals = []
    x = np.asarray(train_data)
    y = np.asarray(train_labels)
    model.fit(x, y)

    # Store trained weights
    log.info(f"Trained weights: {model.weights}")

    # score classifier

    train_accuracy = model.score(x, y)
    log.info(f"Accuracy from the train data : {np.round(100 * train_accuracy, 2)}%")

    # Predict on test data
    y_predict = model.predict(test_data)
    if args['mode'] == 'regression':
        y_predict = [ x[0] for x in y_predict]

    # Score Test Prediction
    x = np.asarray(test_data)
    y = np.asarray(test_labels)
    test_accuracy = model.score(x, y)
    log.info(f"Accuracy from the test data : {np.round(100 * test_accuracy, 2)}%")

    # output results
    res = {"train_scores": train_accuracy,
           "test_scores": test_accuracy,
           "test_pred": y_predict,
           "test_labels": test_labels,
           "num_ansatz_parameters":len(ansatz.parameters),
           "objective_func_vals": objective_func_vals,
           "args": args
           }
    pickle.dump(res, open(os.path.join(mylib.output, args['file_output']), 'wb'))

    log.info(f"RESULTS: {res}")
    if args['mode'] == 'classification':
        # Convert predictions from -1/1 to 0/1 for binary classification
        y_pred_flat = y_predict.flatten() if hasattr(y_predict, 'flatten') else y_predict
        y_pred_binary = np.where(np.array(y_pred_flat) == -1, 0, 1)
        log.info(f"\n{classification_report(test_labels, y_pred_binary, target_names=['Low', 'High'])}")
    elif args['mode'] == 'regression':
        log.info(f"\nR2={r2_score(test_labels, y_predict)}; MSE= {mean_squared_error(test_labels, y_predict)}; Pearson={pearsonr(test_labels, y_predict)}")


if __name__ == "__main__":
    main()
