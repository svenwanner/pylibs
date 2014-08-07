# -*- coding: utf-8 -*-

import vigra
import numpy as np
import scipy as sp

from mypy.tools.cg import transformations as trans


def compute_camera_trail(initial_position, euler_angles, number_of_sampling_points, baseline, translation_vector):

    """Return dictionary with camera index and spacial position of the camera
    >>> initial_position = np.array([2.63167, -5.28107, 4.63259])
    >>> euler_angles = np.array([2.63167, -5.28107, 4.63259])
    >>> number_of_sampling_points = 231
    >>> baseline = 0.0153837881352217
    >>> translation_vector = np.array[(2.4268112182617188, 2.574542999267578, -0.038271427154541016)]
    >>> trail = compute_camera_trail(initial_position, euler_angles, number_of_sampling_points, baseline, translation_vector)
    >>> positions = np.zeros((3,3))
    >>> positions[0,0] = trail[1][0]; positions[0,1] = trail[1][1]; positions[0,2] = trail[1][2]
    >>> positions[1,0] = trail[2][0]; positions[1,1] = trail[2][1]; positions[1,2] = trail[2][2]
    >>> positions[2,0] = trail[230][0]; positions[2,1] = trail[230][1]; positions[2,2] = trail[230][2]
    >>> gt = np.array([[2.6422219276428223, -5.269871711730957, 4.632420063018799], [2.652773141860962, -5.258677959442139, 4.632253646850586], [5.058481693267822, -2.7065224647521973, 4.594315052032471]])
    >>> numpy.allclose(positions, gt)
    """

    assert isinstance(initial_position, np.ndarray)
    assert len(initial_position.shape[0]) == 3
    assert isinstance(euler_angles, np.ndarray)
    assert len(euler_angles.shape[0]) == 3
    assert isinstance(number_of_sampling_points, int)
    assert isinstance(baseline, float)
    assert isinstance(translation_vector, np.ndarray)
    assert len(translation_vector.shape[0]) == 3



    pass





if __name__ == "__main__":
    import doctest
    import random
    np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()



# from mypy.tools.cg import _transformations as ftrans
# vx = np.array([1, 0, 0])
# vy = np.array([0, 1, 0])
# vz = np.array([0, 0, 1])
# v0 = np.array([1, 1, 1])
# v1 = np.array([-1, -1, -1])
#
# print ftrans.identity_matrix()
# print "\nrandom_vector(10):", ftrans.random_vector(10)
# print "\nvector_product(vx, vy):", trans.vector_product(vx, vy)
# print "\nangle_between_vectors(vx, vy):", trans.angle_between_vectors(vx, vy)
# unit = trans.unit_vector(v0)
# print "\nunit_vector(v0):", unit
# print "\nnp.sqrt(np.sum(unit[:]**2)):", np.sqrt(np.sum(unit[:]**2))
# print "\nvector_norm(v0):", trans.vector_norm(v0)
# print "\nrandom_rotation_matrix():", trans.random_rotation_matrix()
# print "\nrandom_quaternion():", trans.random_quaternion()
# print "\nquaternion_matrix(quaternion):", trans.quaternion_matrix(trans.random_quaternion())
# quaternion = trans.quaternion_from_euler(np.pi/2.0, 0.0, np.pi/2.0, axes='sxyz')
# print "\nquaternion_from_euler(ai, aj, ak, axes='sxyz'):", quaternion
# print "\neuler_from_quaternion(quaternion, axes='sxyz'):", trans.euler_from_quaternion(quaternion, axes='sxyz')
# matrix = trans.euler_matrix(np.pi/2.0, 0.0, np.pi/2.0, axes='sxyz')
# print "\neuler_matrix(ai, aj, ak, axes='sxyz'):", matrix
#
# alpha, beta, gamma = np.pi/2.0, 0, np.pi/2.0
# origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
# I = trans.identity_matrix()
# Rx = trans.rotation_matrix(alpha, xaxis)
# Ry = trans.rotation_matrix(beta, yaxis)
# Rz = trans.rotation_matrix(gamma, zaxis)
# R = trans.concatenate_matrices(Rx, Ry, Rz)
# euler = trans.euler_from_matrix(R, 'rxyz')
# print "\n", R
# print "\n", euler