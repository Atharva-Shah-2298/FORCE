import numpy as np
from dipy.sims.voxel import all_tensor_evecs

def bingham_to_sf(f0, k1, k2 , major_axis, minor_axis, vertices):
    """
    Evaluate Bingham distribution for a sphere.

    Parameters
    ----------
    f0: float
        Maximum amplitude of the distribution.
    k1: float
        Concentration along major axis.
    k2: float
        Concentration along minor axis.
    vertices: ndarray (N, 3)
        Unit sphere directions along which the distribution
        is evaluated.
    """

    sf = f0*np.exp(-k1*vertices.dot(major_axis)**2
                   -k2*vertices.dot(minor_axis)**2)
    return sf.T

def bingham_dictionary(target_sphere, odi_list):
    """
    Generate a dictionary of Bingham spherical functions for a given target
    sphere and a list of ODI values.

    Parameters
    ----------
    target_sphere: ndarray (N, 3)
        Unit sphere vertices.
    odi_list: list
        List of ODI values.

    Returns
    -------
    bingham_sf: dict
        Dictionary containing the Bingham spherical functions for each
        vertex in the target sphere and each ODI value.
    """

    bingham_sf = {}
    for i in range(len(target_sphere)):
        vertex_key = tuple(target_sphere[i])
        bingham_sf[i] = {}

        for odi in odi_list:
            k = 1 / np.tan(np.pi / 2 * odi)  # Calculate k based on odi
            evecs = all_tensor_evecs(vertex_key)  # Get eigenvectors for this vertex
            major_axis, minor_axis = evecs[:, 1], evecs[:, 2]

            # Call your function to calculate the Bingham spherical function and store it in the dictionary
            bingham_sf[i][odi] = bingham_to_sf(1, k, k, major_axis, minor_axis, target_sphere)

    return bingham_sf