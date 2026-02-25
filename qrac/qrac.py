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

from typing import List, Dict, Union
from functools import reduce
import numpy as np

from qiskit import QuantumCircuit, Aer, execute
from qiskit.opflow import (
    CircuitOp,
    Zero,
    One,
    CircuitStateFn,
)


def z_to_31p_qrac_basis_circuit(basis: List[int]) -> QuantumCircuit:
    """Return the basis rotation corresponding to the (3,1,p)-QRAC
    Args:
        basis: 0, 1, 2, or 3 for each qubit
    Returns:
        The ``QuantumCircuit`` implementing the rotation.
    """
    circ = QuantumCircuit(len(basis))
    BETA = np.arccos(1 / np.sqrt(3))
    for i, base in enumerate(reversed(basis)):
        if base == 0:
            circ.r(-BETA, -np.pi / 4, i)
            print(-BETA, -np.pi / 4)
        elif base == 1:
            circ.r(np.pi - BETA, np.pi / 4, i)
            print(np.pi - BETA, np.pi / 4)
        elif base == 2:
            circ.r(np.pi + BETA, np.pi / 4, i)
            print(np.pi + BETA, np.pi / 4)
        elif base == 3:
            circ.r(BETA, -np.pi / 4, i)
            print(BETA, -np.pi / 4)
        else:
            raise ValueError(f"Unknown base: {base}")
    return circ


def z_to_21p_qrac_basis_circuit(basis: List[int]) -> QuantumCircuit:
    """Return the basis rotation corresponding to the (2,1,p)-QRAC
    Args:
        basis: 0 or 1 for each qubit
    Returns:
        The ``QuantumCircuit`` implementing the rotation.
    """
    circ = QuantumCircuit(len(basis))
    for i, base in enumerate(reversed(basis)):
        if base == 0:
            circ.r(-1 * np.pi / 4, -np.pi / 2, i)
        elif base == 1:
            circ.r(-3 * np.pi / 4, -np.pi / 2, i)
        else:
            raise ValueError(f"Unknown base: {base}")
    return circ


def qrac_state_prep_1q(*m: int) -> CircuitStateFn:
    """Prepare a single qubit QRAC state
      This function accepts 1, 2, or 3 arguments, in which case it generates a
      1-QRAC, 2-QRAC, or 3-QRAC, respectively.
    Args:
        m: The data to be encoded. Each argument must be 0 or 1.
    Returns:
        The circuit state function.
    """
    if len(m) not in (1, 2, 3):
        raise TypeError(
            f"qrac_state_prep_1q requires 1, 2, or 3 arguments, not {len(m)}."
        )
    if not all(mi in (0, 1) for mi in m):
        raise ValueError("Each argument to qrac_state_prep_1q must be 0 or 1.")

    if len(m) == 3:
        # Prepare (3,1,p)-qrac

        # In the following lines, the input bits are XOR'd to match the
        # conventions used in the paper.

        # To understand why this transformation happens,
        # observe that the two states that define each magic basis
        # correspond to the same bitstrings but with a global bitflip.

        # Thus the three bits of information we use to construct these states are:
        # c0,c1 : two bits to pick one of four magic bases
        # c2: one bit to indicate which magic basis projector we are interested in.

        c0 = m[0] ^ m[1] ^ m[2]
        c1 = m[1] ^ m[2]
        c2 = m[0] ^ m[2]

        base = [2 * c1 + c2]
        cob = z_to_31p_qrac_basis_circuit(base)
        # This is a convention chosen to be consistent with https://arxiv.org/pdf/2111.03167v2.pdf
        # See SI:4 second paragraph and observe that π+ = |0X0|, π- = |1X1|
        sf = One if (c0) else Zero
        # Apply the z_to_magic_basis circuit to either |0> or |1>
        logical = CircuitOp(cob) @ sf
    elif len(m) == 2:
        # Prepare (2,1,p)-qrac
        # (00,01) or (10,11)
        c0 = m[0]
        # (00,11) or (01,10)
        c1 = m[0] ^ m[1]

        base = [c1]
        cob = z_to_21p_qrac_basis_circuit(base)
        # This is a convention chosen to be consistent with https://arxiv.org/pdf/2111.03167v2.pdf
        # See SI:4 second paragraph and observe that π+ = |0X0|, π- = |1X1|
        sf = One if (c0) else Zero
        # Apply the z_to_magic_basis circuit to either |0> or |1>
        logical = CircuitOp(cob) @ sf
    else:
        assert len(m) == 1
        c0 = m[0]
        sf = One if (c0) else Zero

        logical = sf

    return logical.to_circuit_op()


def qrac_state_prep_multiqubit(
    dvars: Union[Dict[int, int], List[int]],
    q2vars: List[List[int]],
    max_vars_per_qubit: int,
) -> CircuitStateFn:
    """
    Prepare a multiqubit QRAC state.
    Args:
        dvars: state of each decision variable (0 or 1)
    """
    remaining_dvars = set(dvars if isinstance(dvars, dict) else range(len(dvars)))

    ordered_bits = []
    for qi_vars in q2vars:
        if len(qi_vars) > max_vars_per_qubit:
            raise ValueError(
                "Each qubit is expected to be associated with at most "
                f"`max_vars_per_qubit` ({max_vars_per_qubit}) variables, "
                f"not {len(qi_vars)} variables."
            )
        if not qi_vars:
            # This probably actually doesn't cause any issues, but why support
            # it (and test this edge case) if we don't have to?
            raise ValueError(
                "There is a qubit without any decision variables assigned to it."
            )
        qi_bits: List[int] = []
        for dv in qi_vars:
            try:
                qi_bits.append(dvars[dv])
            except (KeyError, IndexError):
                raise ValueError(
                    f"Decision variable not included in dvars: {dv}"
                ) from None
            try:
                remaining_dvars.remove(dv)
            except KeyError:
                raise ValueError(
                    f"Unused decision variable(s) in dvars: {remaining_dvars}"
                ) from None

        # Pad with zeros if there are fewer than `max_vars_per_qubit`.
        # NOTE: This results in everything being encoded as an n-QRAC,
        # even if there are fewer than n decision variables encoded in the qubit.
        # In the future, we plan to make the encoding "adaptive" so that the
        # optimal encoding is used on each qubit, based on the number of
        # decision variables assigned to that specific qubit.
        # However, we cannot do this until magic state rounding supports 2-QRACs.
        while len(qi_bits) < max_vars_per_qubit:
            qi_bits.append(0)

        ordered_bits.append(qi_bits)

    if remaining_dvars:
        raise ValueError(f"Not all dvars were included in q2vars: {remaining_dvars}")

    qracs = [qrac_state_prep_1q(*qi_bits) for qi_bits in ordered_bits]
    logical = reduce(lambda x, y: x ^ y, qracs)
    return logical


def test_functions():
    backend = Aer.get_backend('statevector_simulator')

    # QRAC(1,1)
    #circuit = qrac_state_prep_1q(1)
    #print(circuit)

    # QRAC(2,1)
    #circuit = qrac_state_prep_1q(0, 0).to_circuit()
    #print(circuit)
    #result = execute(circuit, backend=backend).result()
    #print(result.get_statevector().data)

    # QRAC(3,1)
    circuit = qrac_state_prep_1q(1, 1, 0).to_circuit()
    print(circuit)
    result = execute(circuit, backend=backend).result()
    print(result.get_statevector().data)

    # MULTI-QUBIT
    # Each data point will be "dvars", i.e. a list of binary decision variables
    # Group them together to map them onto a single qubit using q2vars
    # n = max_vars_per_qubit is at most 3, because (4,1,p)-QRAC does not exist
    # Note that everything is encoded as n-QRAC, please read the note above

    #dvars = [0, 1, 1, 0, 1, 1]
    #q2vars = [[0, 1], [2, 3, 4], [5]]
    #max_vars_per_qubit = 3

    #dvars = [0, 1]
    #q2vars = [[0], [1]]
    #max_vars_per_qubit = 1

    #dvars = [0, 0]
    #q2vars = [[0, 1]]
    #max_vars_per_qubit = 2

    #circuit = qrac_state_prep_multiqubit(dvars, q2vars, max_vars_per_qubit)
    #print(circuit)


test_functions()



