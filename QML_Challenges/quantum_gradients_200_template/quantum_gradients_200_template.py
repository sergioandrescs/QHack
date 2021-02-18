#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #

    # Gradient Calculation
    import math

    in_place_calculation = circuit(weights)
    half_pi_shifts = []

    def parameter_shift_term(qnode, params, shift_value, index):
        shifted = params.copy()
        shifted[index] += shift_value
        forward = qnode(shifted)  # forward evaluation
        half_pi_shifts.append(forward)

        shifted[index] -= shift_value*2
        backward = qnode(shifted)  # backward evaluation
        half_pi_shifts.append(backward)

        return (forward - backward)/(2*math.sin(shift_value))

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

    def parameter_hessian_term(qnode, params, shift_value, row, col):
        # Reference for future review: https://arxiv.org/pdf/2008.06517.pdf
        # To optimize, pi/2 and pi/4 as shift values were considered.
        # DO NOT FORGET
        if row == col and shift_value == math.pi/2:
            shifted = params.copy()
            shifted[row] += math.pi

            forward = half_pi_shifts[row*2]
            backward = half_pi_shifts[row*2 + 1]

            return (forward - (2*in_place_calculation) + backward)/2
        else:
            step_1 = qnode(shifter(params, shift_value, row, col, [1, 1]))
            step_2 = qnode(shifter(params, shift_value, row, col, [-1, 1]))
            step_3 = qnode(shifter(params, shift_value, row, col, [1, -1]))
            step_4 = qnode(shifter(params, shift_value, row, col, [-1, -1]))

            return (step_1 - step_2 - step_3 + step_4)/pow(2*math.sin(shift_value), 2)

    for i in range(len(weights)):
        gradient[i] = parameter_shift_term(circuit, weights, math.pi/2, i)

    for i in range(len(weights)):
        for j in range(i, len(weights)):
            hessian[i][j] = parameter_hessian_term(
                circuit, weights, math.pi/2, i, j)
            hessian[j][i] = hessian[i][j]

    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
