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
from qiskit.visualization import plot_bloch_vector


x000 = np.array([0.88807383+0.j,         0.32505758+0.32505758j])
theta000 = 2 * np.arccos(0.88807383)
phi000 = np.arccos(0.32505758 / np.sqrt(1 - 0.88807383**2))

x001 = np.array([0.45970084+0.j,         0.62796303+0.62796303j])
theta001 = 2 * np.arccos(0.45970084)
phi001 = np.arccos(0.62796303 / np.sqrt(1 - 0.45970084**2))

x010 = np.array([0.88807383+0.j,         -0.32505758+0.32505758j])
theta010 = 2 * np.arccos(0.88807383)
phi010 = np.arccos(0.32505758 / np.sqrt(1 - 0.88807383**2)) + (np.pi / 2)

x011 = np.array([0.45970084+0.j,         -0.62796303+0.62796303j])
theta011 = 2 * np.arccos(0.45970084)
phi011 = np.arccos(0.62796303 / np.sqrt(1 - 0.45970084**2)) + (np.pi / 2)


x100 = np.array([0.88807383+0.j,         0.32505758-0.32505758j])
theta100 = 2 * np.arccos(0.88807383)
phi100 = np.arccos(0.32505758 / np.sqrt(1 - 0.88807383**2)) + (3 * np.pi / 2)

x101 = np.array([0.45970084+0.j,         0.62796303-0.62796303j])
theta101 = 2 * np.arccos(0.45970084)
phi101 = np.arccos(0.62796303 / np.sqrt(1 - 0.45970084**2)) + (3 * np.pi / 2)

x110 = np.array([0.88807383+0.j,         -0.32505758-0.32505758j])
theta110 = 2 * np.arccos(0.88807383)
phi110 = np.arccos(0.32505758 / np.sqrt(1 - 0.88807383**2)) + np.pi

x111 = np.array([0.45970084+0.j,         -0.62796303-0.62796303j])
theta111 = 2 * np.arccos(0.45970084)
phi111 = np.arccos(0.62796303 / np.sqrt(1 - 0.45970084**2)) + np.pi

fig = plot_bloch_vector([1, theta000, phi000], coord_type='spherical')
fig.savefig('bloch.png')


'''
# From qrac.py
x000 = np.array([0.88807383+0.j,              0.32505758+0.32505758j])
x001 = np.array([0.32505758-3.25057584e-01j,  0.88807383-2.22044605e-16j])
x010 = np.array([-0.62796303-6.27963030e-01j, -0.45970084+2.77555756e-17j])
x011 = np.array([0.45970084+0.j,              0.62796303-0.62796303j])
x100 = np.array([-0.62796303-6.27963030e-01j, 0.45970084-1.11022302e-16j])
x101 = np.array([-0.45970084+0.j,             0.62796303-0.62796303j])
x110 = np.array([0.88807383+0.j,              -0.32505758-0.32505758j])
x111 = np.array([-0.32505758+0.32505758j,     0.88807383+0.j        ])
'''
