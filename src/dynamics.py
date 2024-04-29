from numpy.random import randint, random
from utils import get_energy_of_spin, get_change_in_E, get_prob_of_flipping, are_nn, select_random_coordinate, select_two_random_coordinates

def change_spin_glauber(spin_array, lx, ly, j, k_B, T):

    itrial, jtrial = select_random_coordinate(spin_array)
    E = get_energy_of_spin(itrial, jtrial, spin_array, j)
    delta_E = get_change_in_E(E)
    # flag corresponding to metropolis test: change spin if it
    # lowers the energy if (ΔE < 0) or with prob exp(ΔE / (k_b)T)
    metropolis_flag = random() <= get_prob_of_flipping(delta_E, k_B, T) 
    if delta_E < 0 or metropolis_flag:
        spin_array[itrial, jtrial] *= -1
        return delta_E 
    return

def change_spin_kawasaki(spin_array, lx, ly, j, k_B, t):
    # choose two spins randomly
    spin1_coordinate, spin2_coordinate = select_two_random_coordinates(spin_array)
    spin1 = spin_array[spin1_coordinate]
    spin2 = spin_array[spin2_coordinate]
    same_direction_flag = spin1 == spin2

    while (same_direction_flag):
        spin1_coordinate, spin2_coordinate = select_two_random_coordinates(spin_array)
        spin1 = spin_array[spin1_coordinate]
        spin2 = spin_array[spin2_coordinate]
        same_direction_flag = spin1 == spin2

    # Add (-1) because we are flipping the spins to compute the change in energy
    E_spin1 = get_energy_of_spin(*spin1_coordinate, spin_array, j)
    E_spin2 = get_energy_of_spin(*spin2_coordinate, spin_array, j)
    delta_E = get_change_in_E(E_spin1) + get_change_in_E(E_spin2) 
    if (are_nn(spin1_coordinate, spin2_coordinate, lx, ly)):
        delta_E += 4
    
    metropolis_flag = random() <= get_prob_of_flipping(delta_E, k_B, t) 
    if delta_E <= 0 or metropolis_flag:
        spin_array[spin1_coordinate] *= -1
        spin_array[spin2_coordinate] *= -1
        return delta_E
    return