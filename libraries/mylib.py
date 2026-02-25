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

from datetime import datetime
import pandas as pd
import os, re
import sklearn.metrics
import scipy.stats
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from comut import comut
import matplotlib
import palettable

from tensorflow.keras.utils import to_categorical
import category_encoders as ce
import numpy as np
import pandas as pd

import networkx as nx
import multiprocessing as mp
from itertools import combinations


def path_worker(inqueue, G, output, sentinel, path_length, cost_values, qbit_props, mid_path):
    result = []
    count = 0
    for pair in iter(inqueue.get, sentinel):
        source, target = pair
        for path in nx.all_simple_paths(G, source = source, target = target,
                                        cutoff = path_length-1):
            if len(path) == path_length:
                cv = []
                cv.append( path)
                for cost_value in cost_values:
                    cv.append( np.median(qbit_props[ qbit_props.qubit.isin(path) ][cost_value]) )
                    cv.append( np.median(qbit_props[ qbit_props.qubit.isin([ path[x] for x in mid_path]) ][cost_value]) )
                    cv.append( np.max(qbit_props[ qbit_props.qubit.isin(path) ][cost_value]) )
                result.append(cv)
                count += 1
    output.put(result)

def get_paths_parallel(G, nodes, output, sentinel, path_length, cost_values, qbit_props, mid_path):
    result = []
    inqueue = mp.Queue()
    for items in combinations(nodes, r=2): 
        source = items[0]
        target = items[1]
        inqueue.put((source, target))
    procs = [mp.Process(target = path_worker, args = (inqueue, G, output, sentinel, path_length, cost_values, qbit_props, mid_path))
             for i in range(mp.cpu_count())]
    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:    
        inqueue.put(sentinel)
    for proc in procs:
        result.extend(output.get())
    for proc in procs:
        proc.join()
    return result

def get_qubit_path( backend, path_length, readoutThreshold = 0.12, edgeThreshold = 0.2, save = False ):
    coupling_map = np.array(backend.coupling_map.get_edges())

    # Get qubit properites
    qbit_props = []
    for i in range(backend.num_qubits):
        prop = backend.qubit_properties(i)
        measure_prop = backend.target["measure"][(i,)]
        qbit_props.append( [i, measure_prop.error, prop.t1, prop.t2, prop.frequency ] )
    qbit_props = pd.DataFrame( qbit_props, columns = [ 'qubit', 'readout', 't1', 't2', 'frequency'] )
    qbit_props.head()
    
    # Get edge connection properties
    edge_props = []
    for i in range(len(coupling_map)):
        edge = ( coupling_map[i][0], coupling_map[i][1] )
        if 'ecr' in backend.target.operations:
            edge_props.append( [edge, backend.target["ecr"][edge].error] )
        elif 'cz' in backend.target.operations:
            edge_props.append( [edge, backend.target["cz"][edge].error] )

    edge_props = pd.DataFrame( edge_props, columns = [ 'edge', 'error'] )
    edge_props.head()

    cost_values = [ 'readout', 't1', 't2']
    center = int(np.floor(path_length/2))
    mid_path = list(range(center-1, center+3))

    # generate graph and remove low quality nodes and edges
    G = nx.from_edgelist(coupling_map)
    nodes_to_remove = qbit_props[ qbit_props.readout >= readoutThreshold ].qubit.to_list()
    edges_to_remove = edge_props[ edge_props.error >= edgeThreshold ].edge.to_list()
    G.remove_nodes_from(nodes_to_remove)
    G.remove_edges_from(edges_to_remove)
    
    # Get all paths of size path_length between all pairs of nodes
    output = mp.Queue()
    sentinel = None
    all_nodes = list(G.nodes)
    allpaths = get_paths_parallel(G, all_nodes, output, sentinel, path_length, cost_values, qbit_props, mid_path)
    path_cost_df = pd.DataFrame( allpaths, columns = ['path'] + [ y + x for y in cost_values for x in ['','_mid', '_max']])
    path_cost_df['readout_average_totalAndMid'] = path_cost_df[['readout','readout_mid']].mean(axis = 1)
    path_cost_df['readout_average_totalAndMidandMax'] = path_cost_df[['readout','readout_mid']].mean(axis = 1)
    path_cost_df = path_cost_df[list(path_cost_df.columns[1:]) + [path_cost_df.columns[0]]]
    path_cost_df = path_cost_df.sort_values(cost_values[0], ascending = True)
    if save:
        path_cost_df.to_csv( 'PathCosts_' + str(path_length) + '_' + backend.name + 
                            '_drop' + str(readoutThreshold) + '_edgedrop' + str(edgeThreshold) + '.csv', index = False ) 

    # Select path by minimum of sum of Ranks
    # path_cost_df_rnk = path_cost_df[['readout', 'readout_mid',]].rank()
    # path_cost_df_rnk['sum']  = path_cost_df_rnk.sum(axis = 1)
    # path_cost_df_rnk = path_cost_df_rnk.sort_values( 'sum', ascending = True)
    # select_path = path_cost_df.loc[path_cost_df_rnk.index[0],'path']

    # Select path by filtering for top 5% readout error and then select smallest readout_mid
    path_cost_dff = path_cost_df.sort_values('readout', ascending=True).iloc[0:int(np.floor(len(path_cost_df)*0.05)),:]
    select_path= path_cost_dff.sort_values('readout_mid', ascending=True).path.iloc[0]

    return list(select_path), path_cost_df




def data_encoder(args, train_data, test_data, num_class, num_motifs):
    
    if args['encoder'] == 'one-hot':
        #jie
        motifs = [item for item in args['motifs_to_use'] if 'motif' in item]
        non_motifs = [item for item in args['motifs_to_use'] if 'motif' not in item]
        #encoder = ce.BinaryEncoder()
        
        train_encoded = pd.DataFrame()
        test_encoded = pd.DataFrame()

        if motifs:
            motif_train_data = to_categorical(train_data[motifs], num_classes=num_class)
            motif_test_data = to_categorical(test_data[motifs], num_classes=num_class)
            
            motif_train_data = motif_train_data.reshape(motif_train_data.shape[0], motif_train_data.shape[1] * motif_train_data.shape[2])
            motif_test_data = motif_test_data.reshape(motif_test_data.shape[0], motif_test_data.shape[1] * motif_test_data.shape[2])
            motif_train_df = pd.DataFrame(motif_train_data,columns=motifs)
            motif_test_df = pd.DataFrame(motif_test_data,columns=motifs)

            train_encoded = pd.concat([train_encoded, motifs_train_df], axis=1)
            test_encoded = pd.concat([test_encoded, motifs_test_df], axis=1)              

        for motif_name in non_motifs:
            encoder = ce.OneHotEncoder(cols=[motif_name])
            
            train_column = train_data[motif_name]
            test_column = test_data[motif_name]
            
            base_array = np.unique(np.concatenate([train_column, test_column]))
            base = pd.DataFrame(base_array, columns=[motif_name]).astype('category')
            
            if len(base) == 2:
                # if there are only 2 values, it should be encoded in 1 bit but Binary encoder will do it in 2
                value_map = {val: idx for idx, val in enumerate(base_array)}
                
                train_column = train_column.map(value_map)
                test_column = test_column.map(value_map)
                
                train_column = pd.DataFrame(train_column, columns=[motif_name])
                test_column = pd.DataFrame(test_column, columns=[motif_name])
            else:
                encoder.fit(base)
                train_column = encoder.transform(train_column.astype('category'))
                test_column = encoder.transform(test_column.astype('category'))

            train_encoded = pd.concat([train_encoded, train_column], axis=1)
            test_encoded = pd.concat([test_encoded, test_column], axis=1)
            
        train_data = train_encoded.values
        test_data = test_encoded.values                
        
    elif args['encoder'] == 'binary':

        #jie
        motifs = [item for item in args['motifs_to_use'] if 'motif' in item]
        non_motifs = [item for item in args['motifs_to_use'] if 'motif' not in item]
        #encoder = ce.BinaryEncoder()

        train_encoded = pd.DataFrame()
        test_encoded = pd.DataFrame()


        if motifs:
            motifs_train_data = train_data[motifs]
            motifs_test_data = test_data[motifs]
            encoder = ce.BinaryEncoder()
            base_array = np.unique(np.concatenate([motifs_train_data, motifs_test_data]))
            
            base = pd.DataFrame(base_array).astype('category')
            base.columns = [motifs[0]]
            for motif_name in motifs[1:]:
                base[motif_name] = base.loc[:, motifs[0]]
            encoder.fit(base)

            motifs_train_data = encoder.transform(motifs_train_data.astype('category'))
            motifs_test_data = encoder.transform(motifs_test_data.astype('category'))

            train_encoded = pd.concat([train_encoded, motifs_train_data], axis=1)
            test_encoded = pd.concat([test_encoded, motifs_test_data], axis=1)            
            

        for motif_name in non_motifs:
            encoder = ce.BinaryEncoder(cols=[motif_name])
            
            train_column = train_data[motif_name]
            test_column = test_data[motif_name]
            
            base_array = np.unique(np.concatenate([train_column, test_column]))
            base = pd.DataFrame(base_array, columns=[motif_name]).astype('category')
            
            if len(base) == 2:
                # if there are only 2 values, it should be encoded in 1 bit but Binary encoder will do it in 2
                value_map = {val: idx for idx, val in enumerate(base_array)}
                
                train_column = train_column.map(value_map)
                test_column = test_column.map(value_map)
                
                train_column = pd.DataFrame(train_column, columns=[motif_name])
                test_column = pd.DataFrame(test_column, columns=[motif_name])
            else:
                encoder.fit(base)
                train_column = encoder.transform(train_column.astype('category'))
                test_column = encoder.transform(test_column.astype('category'))

            train_encoded = pd.concat([train_encoded, train_column], axis=1)
            test_encoded = pd.concat([test_encoded, test_column], axis=1)

        train_data = train_encoded.values
        test_data = test_encoded.values            

            
    elif args['encoder'] == 'binary-flip':

        #jie
        motifs = [item for item in args['motifs_to_use'] if 'motif' in item]
        non_motifs = [item for item in args['motifs_to_use'] if 'motif' not in item]
        #encoder = ce.BinaryEncoder()

        train_encoded = pd.DataFrame()
        test_encoded = pd.DataFrame()


        if motifs:
            motifs_train_data = train_data[motifs]
            motifs_test_data = test_data[motifs]
            encoder = ce.BinaryEncoder()
            base_array = np.unique(np.concatenate([motifs_train_data, motifs_test_data]))
            
            base = pd.DataFrame(base_array).astype('category')
            base.columns = [motifs[0]]
            for motif_name in motifs[1:]:
                base[motif_name] = base.loc[:, motifs[0]]
            encoder.fit(base)

            motifs_train_data = encoder.transform(motifs_train_data.astype('category'))
            motifs_test_data = encoder.transform(motifs_test_data.astype('category'))

            train_encoded = pd.concat([train_encoded, motifs_train_data], axis=1)
            test_encoded = pd.concat([test_encoded, motifs_test_data], axis=1)            
            

        for motif_name in non_motifs:
            encoder = ce.BinaryEncoder(cols=[motif_name])
            
            train_column = train_data[motif_name]
            test_column = test_data[motif_name]
            
            base_array = np.unique(np.concatenate([train_column, test_column]))
            base = pd.DataFrame(base_array, columns=[motif_name]).astype('category')
            
            if len(base) == 2:
                # if there are only 2 values, it should be encoded in 1 bit but Binary encoder will do it in 2
                value_map = {val: idx for idx, val in enumerate(base_array)}
                
                train_column = train_column.map(value_map)
                test_column = test_column.map(value_map)
                
                train_column = pd.DataFrame(train_column, columns=[motif_name])
                test_column = pd.DataFrame(test_column, columns=[motif_name])
            else:
                encoder.fit(base)
                train_column = encoder.transform(train_column.astype('category'))
                test_column = encoder.transform(test_column.astype('category'))

            train_encoded = pd.concat([train_encoded, train_column], axis=1)
            test_encoded = pd.concat([test_encoded, test_column], axis=1)

        train_data = train_encoded.replace({0: 1, 1: 0})
        test_data = test_encoded.replace({0: 1, 1: 0})

        train_data = train_data.values
        test_data = test_data.values            

    else:
        raise ValueError('Invalid encoding type.')
    
    return train_data, test_data        

def get_time():
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S')

# Scale array between specified range
def scaler( m, tmin = -1, tmax = 1):
    return (m-np.min(m))/(np.max(m)-np.min(m)) * (tmax - tmin) + tmin

def preprocess_data(dir_root, args):
    train_data = pd.read_csv(os.path.join(dir_root, args['file_train_data']), encoding='unicode_escape', sep=',')
    test_data = pd.read_csv(os.path.join(dir_root, args['file_test_data']), encoding='unicode_escape', sep=',')
    
    # Get the labels
    train_labels = np.array(train_data[args['label_name']])
    test_labels = np.array(test_data[args['label_name']])

    # if classification task and if there there is threshold to binarize, do so to the labels, else if regressio then normalize the value to -1 to 1
    if args['mode'] == 'classification':
        if args['label_binarization_threshold'] != None:
            train_labels[train_labels > args['label_binarization_threshold']] = 1
            train_labels[train_labels < 1] = args['min_label_value']
            test_labels[test_labels > args['label_binarization_threshold']] = 1
            test_labels[test_labels < 1] = args['min_label_value']
    elif args['mode'] == 'regression':
        train_labels = scaler( train_labels)        
        test_labels = scaler( test_labels)
        
    # reduce data to just the motifs of interest
    train_data = train_data[args['motifs_to_use']]
    test_data = test_data[args['motifs_to_use']]

    # get the class and motif coounts
    min_class = np.min(np.unique(np.concatenate([train_data, test_data])))
    max_class = np.max(np.unique(np.concatenate([train_data, test_data])))
    num_class = max_class - min_class + 1
    num_motifs = len(args['motifs_to_use'])

    train_data = train_data - min_class
    test_data = test_data - min_class
    return train_data, test_data, train_labels, test_labels, num_class, num_motifs
