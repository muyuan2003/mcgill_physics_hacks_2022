import set_up as s
import numpy as np
from scipy import interpolate


max_iter = None

def partial_derivative_x(field):
    diff = np.zeros_like(field)

    diff[1:-1, 1:-1] = ( (field[2:, 1:-1] - field[0:-2, 1:-1])
                        / (2 * s.element_length) )

    return diff


def partial_derivative_y(field):
    diff = np.zeros_like(field)

    diff[1:-1, 1:-1] = ( (field[1:-1, 2:] - field[1:-1, 0:-2])
                            / ( 2 * s.element_length ) )

    return diff


def laplace(field):
    diff = np.zeros_like(field)

    diff[1:-1, 1:-1] = ( field[0:-2, 1:-1] + field[1:-1, 0:-2] - 4 * field[1:-1, 1:-1] + field[2:, 1:-1] + field[1:-1, 2:] ) \
                            / ( s.element_length ** 2 )


    return diff


def divergence(vector_field):
    divergence_applied = partial_derivative_x(vector_field[..., 0]) + partial_derivative_y(vector_field[..., 1])

    return divergence_applied


def gradient(field):
    gradient_applied = np.concatenate(
        (partial_derivative_x(field)[..., np.newaxis],
         partial_derivative_y(field)[..., np.newaxis],
        ),
        axis=-1,
    )

    return gradient_applied


def curl_2d(vector_field):
    curl_applied = partial_derivative_x(vector_field[..., 1]) - partial_derivative_y(vector_field[..., 0])

    return curl_applied

def trace_back(vector_field):
    traced_back_position = np.clip(
        ( s.coordinates - s.dt * vector_field ),
        0.0,
        s.GRAPH_MAX_VAL,
    )

    return traced_back_position

def advect(field, vector_field):
    previous_positions = trace_back(vector_field)

    advected_field = interpolate.interpn(
        points=(s.x, s.y),
        values=field,
        xi=previous_positions,
    )

    return advected_field


def diffusion_operator(vector_field_flattened):
    vector_field = vector_field_flattened.reshape(s.vector_shape)

    diffusion_applied = ( vector_field - s.KINEMATIC_VISCOSITY * s.dt * laplace(vector_field) )

    return diffusion_applied.flatten()


def poisson_operator(field_flattened):
    field = field_flattened.reshape(s.scalar_shape)

    poisson_applied = laplace(field)

    return poisson_applied.flatten()