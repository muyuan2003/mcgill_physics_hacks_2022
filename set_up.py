import numpy as np

#  Changeable
GRAPH_MAX_VAL = 1.0
GRID_SIZE = 41
PICTURE_NUM = 100
dt = 0.05
KINEMATIC_VISCOSITY = 0.00001

def decaying_time_function(time):
    return_value = 1.0 - 0.8 * time
    if return_value < 0:
        return 0
    else:
        return return_value

def increasing_time_function(time):
    return 0.1 + 4 * time if time < 2 else 0.0

def create_forces(time, point):
    # Could do an increasing factor too
    # Creates a variable number of independant forces on regions of map 
    # Affects velocity Field
    #time_factor = decaying_time_function(time)
    time_factor = increasing_time_function(time)

    if (point[0] > 0.2) and (point[0] < 0.5) and (point[1] > 0.5) and (point[1] < 0.9):
        return time_factor * np.array([0.2, -2.5])
    elif (point[0] > 0.6) and (point[0] < 0.8) and (point[1] > 0.6) and (point[1] < 0.7):
        return time_factor * np.array([-1.0, 0.5])
    else:
        return time_factor * np.array([0.0, 0.0])



# Don't change
element_length = GRAPH_MAX_VAL / (GRID_SIZE - 1)
scalar_shape = (GRID_SIZE, GRID_SIZE)
vector_shape = (GRID_SIZE, GRID_SIZE, 2)

x = np.linspace(0.0, GRAPH_MAX_VAL, GRID_SIZE)
y = np.linspace(0.0, GRAPH_MAX_VAL, GRID_SIZE)
X, Y = np.meshgrid(x, y, indexing="ij")
coordinates = np.concatenate(
    (
        X[..., np.newaxis],
        Y[..., np.newaxis],
    ), axis=-1,
)

create_forces_on_grid = np.vectorize(
    pyfunc=create_forces,
    signature="(),(d)->(d)",
)

