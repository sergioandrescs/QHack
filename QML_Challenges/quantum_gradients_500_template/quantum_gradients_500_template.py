#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    import math

    # Create qnode for quantum state

    gradient = np.zeros([len(params)], dtype=np.float64)
    behaviour_matrix = np.zeros([len(params), len(params)], dtype=np.float64)

    # Obtain gradient
    def parameter_shift_term(qnode, params, shift_value, index):
        shifted = params.copy()
        shifted[index] += shift_value
        forward = qnode(shifted)  # forward evaluation

        shifted[index] -= shift_value*2
        backward = qnode(shifted)  # backward evaluation

        return (forward - backward)/(2*math.sin(shift_value))

    for i in range(len(params)):
        gradient[i] = parameter_shift_term(qnode, params, math.pi/2, i)

    # print(gradient)

    # Metric tensor

    # First of all, reset state
    dev.reset()

    @qml.qnode(dev)
    def circuit(params):
        variational_circuit(params)
        return qml.state()

    # Execution with original params
    original_params_state = np.conj(circuit(params))
    dev.reset()

    def shifter(params, shift_value, row, col, signs):
        r""" This function executes the shift needed for the Hessian calculation.

        Args:
            row (int): First index of the shift
            col (int): Second index of the shift
            signs (tuple): Tuple with signs (+1,-1) os the shift to be made

        Returns: 
            array: Shifted params 
        """
        shifted = params.copy()
        shifted[row] += shift_value*signs[0]
        shifted[col] += shift_value*signs[1]

        return shifted

    def magnitude_calculator(bra, ket):
        return pow(np.abs(np.inner(bra, ket)), 2)

    def parameter_behave_terms(qnode, params, shift_value, row, col):
        # Reference for future review: https://arxiv.org/pdf/2008.06517.pdf
        dev.reset()
        qnode(shifter(params, shift_value, row, col, [1, 1]))
        step_1 = magnitude_calculator(original_params_state, dev.state)

        dev.reset()
        qnode(shifter(params, shift_value, row, col, [1, -1]))
        step_2 = magnitude_calculator(original_params_state, dev.state)

        dev.reset()
        qnode(shifter(params, shift_value, row, col, [-1, 1]))
        step_3 = magnitude_calculator(original_params_state, dev.state)

        dev.reset()
        qnode(shifter(params, shift_value, row, col, [-1, -1]))
        step_4 = magnitude_calculator(original_params_state, dev.state)

        return (-step_1 + step_2 + step_3 - step_4)/8

    for i in range(len(params)):
        for j in range(i, len(params)):
            behaviour_matrix[i][j] = parameter_behave_terms(
                circuit, params, math.pi/2, i, j)
            behaviour_matrix[j][i] = behaviour_matrix[i][j]

    # print(behaviour_matrix)

    # qml.metric_tenser return a block diagonal matrix.
    # Can be used for comparison, but it is not to be used for this task
    #print(np.round(qml.metric_tensor(qnode)(params), 8))

    inv_behaviour_matrix = np.linalg.inv(behaviour_matrix)
    # print(inv_behaviour_matrix)

    natural_grad = inv_behaviour_matrix.dot(gradient)

    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
