import numpy as np

def select_random_coordinate(spins_array):
    idx = np.random.randint(0, len(spins_array.ravel()))
    coordinate = np.unravel_index(idx, spins_array.shape)
    return coordinate

def select_two_random_coordinates(spin_array):
    return [select_random_coordinate(spin_array) for _ in range(2)]

def get_neighbors_coordinates(i, j, shape):
    rows, cols = shape
    neighbors = []
    
    # Get the indices of the neighboring points
    left = (i, (j - 1) % cols)
    right = (i, (j + 1) % cols)
    top = ((i - 1) % rows, j)
    bottom = ((i + 1) % rows, j)
    
    # Add the neighboring points to the list
    neighbors.extend([left, right, top, bottom])
    
    return neighbors

def get_neighbor_spins(neighbors_coordinates, spins_array):
    return np.array([spins_array[neighbor] for neighbor in neighbors_coordinates])

def get_energy_of_spin(row: int, col: int, spins_array: np.array, J: float) -> int:
    # Get the neighbors of the current spin (row, col)
    neighbors_coordinates = get_neighbors_coordinates(row, col, spins_array.shape)
    # Get the spins of the neighbors
    neighbors_spins = get_neighbor_spins(neighbors_coordinates, spins_array)
    # Compute the energy of the current spin
    return -J*spins_array[row,col]*np.sum(neighbors_spins)

def get_change_in_E(E: int) -> int:
    return -2*E

def get_prob_of_flipping(delta_E: int, k_B: float, T: float) -> float:
    return np.exp(-delta_E / (k_B*T))

def are_nn(spin1_coordinate: tuple, spin2_coordinate: tuple, lx: int, ly: int) -> bool:
    return (
        (spin1_coordinate[0] + 1) % lx == spin2_coordinate[0] and spin1_coordinate[1] == spin2_coordinate[1] or
        (spin1_coordinate[0] - 1) % lx == spin2_coordinate[0] and spin1_coordinate[1] == spin2_coordinate[1] or
        (spin1_coordinate[0] == spin2_coordinate[0] and (spin1_coordinate[1] + 1) % ly == spin2_coordinate[1]) or
        (spin1_coordinate[0] == spin2_coordinate[0] and (spin1_coordinate[1] - 1) % ly == spin2_coordinate[1])
        )