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

from typing import List
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.circuit import ParameterVector


def one_qubit_21p_qrac(basis: List[int]) -> QuantumCircuit:
    assert len(basis) == 2, "Please provide exactly 2 bits."

    qr = QuantumRegister(1, 'q')
    circ = QuantumCircuit(qr)
    x0 = basis[0]
    x1 = basis[1]
    theta = (np.pi / 4) + (np.pi / 2) * (x1 - x0) + np.pi * x1 * x0
    circ.ry(theta, qr[0])

    return circ


def full_circuit_21p_qrac(len_bitstring: int) -> QuantumCircuit:
    assert len_bitstring % 2 == 0, "Length of bitstring should be an even number."
    num_qubits = int(len_bitstring / 2)
    params = ParameterVector('a', length=len_bitstring)

    qr = QuantumRegister(num_qubits, 'q')
    circ = QuantumCircuit(qr)

    for i in range(num_qubits):
        param0 = params[2 * i]
        param1 = params[2 * i + 1]
        theta = (np.pi / 4) + (np.pi / 2) * (param1 - param0) + np.pi * param1 * param0
        circ.ry(theta, qr[i])

    return circ


def one_qubit_31p_qrac(basis: List[int]) -> QuantumCircuit:
    assert len(basis) == 3, "Please provide exactly 3 bits."

    qr = QuantumRegister(1, 'q')
    circ = QuantumCircuit(qr)
    x0 = basis[0]
    x1 = basis[1]
    x2 = basis[2]
    BETA = np.arccos(1 / np.sqrt(3))
    theta = (1 - x2) * BETA + x2 * (np.pi - BETA)
    phi = (np.pi / 4) + (np.pi / 2) * (x1 - x0) + np.pi * x1 * x0
    circ.u(theta, phi, 0, qr[0])

    return circ


def full_circuit_31p_qrac(len_bitstring: int) -> QuantumCircuit:
    assert len_bitstring % 3 == 0, "Length of bitstring should be a multiple of 3."
    num_qubits = int(len_bitstring / 3)
    params = ParameterVector('a', length=len_bitstring)

    qr = QuantumRegister(num_qubits, 'q')
    circ = QuantumCircuit(qr)

    for i in range(num_qubits):
        param0 = params[3 * i]
        param1 = params[3 * i + 1]
        param2 = params[3 * i + 2]

        BETA = np.arccos(1 / np.sqrt(3))
        theta = (1 - param2) * BETA + param2 * (np.pi - BETA)
        phi = (np.pi / 4) + (np.pi / 2) * (param1 - param0) + np.pi * param1 * param0
        circ.u(theta, phi, 0, qr[i])

    return circ


def basis_encoding(len_bitstring: int) -> QuantumCircuit:
    num_qubits = len_bitstring
    params = ParameterVector('a', length=len_bitstring)

    qr = QuantumRegister(num_qubits, 'q')
    circ = QuantumCircuit(qr)

    for i in range(num_qubits):
        param = params[i]
        theta = np.pi * param
        circ.rx(theta, qr[i])

    return circ
