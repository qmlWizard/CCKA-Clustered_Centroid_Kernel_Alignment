import pennylane as qml
from pennylane import numpy as np


def strong_entangled(params, wires):
    qml.StronglyEntanglingLayers(weights = params, wires=wires)

def basic_entangled(params, wires):
    qml.BasicEntanglerLayers(weights = params, wires = wires, rotation = 'RX') 
