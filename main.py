import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

# Input T and C
T = np.array([88.55, 103.12, 113.43, 123.57, 133.32, 153.43, 174.61, 203.59, 244.65, 293.68])
C = np.array([13.9807457094744, 15.8211008539237, 17.0134573150017, 18.0493049690753, 18.9082942099747, 20.6258657172254, 21.5654359831096, 22.8640137442006, 23.8813058126475, 25.6028085272625])

# partial integration of dH
Delta_H = cumtrapz(C, T, initial=0)

# partial integration of dS
Delta_S = cumtrapz(C / T, T, initial=0)

# quadric type interpolation
interp_H = interp1d(T, Delta_H, kind='quadratic', fill_value="extrapolate")
interp_S = interp1d(T, Delta_S, kind='quadratic', fill_value="extrapolate")

# Extrapolation space
T_extended = np.linspace(77, 298, 1000)

# Inter and intrapolation
Delta_H_extended = interp_H(T_extended)
Delta_S_extended = interp_S(T_extended)

# Give value for specific temperature
def calculate_and_display_specific_T(target_T):
    delta_H_specific = interp_H(target_T)
    delta_S_specific = interp_S(target_T)
    return delta_H_specific, delta_S_specific

# Set specific temp.
target_T = 298  # <-----
delta_H_specific, delta_S_specific = calculate_and_display_specific_T(target_T)

plt.figure(figsize=(12, 6))
#dH(T) plott
plt.subplot(1, 2, 1)
plt.plot(T_extended, Delta_H_extended, linestyle='-', color='b', label='ΔH(T)')
plt.scatter(T, Delta_H, color='blue', label='Data Point')

plt.axvline(target_T, color='gray', linestyle='--') #spec
plt.axhline(delta_H_specific, color='gray', linestyle='--') #spec
plt.scatter([target_T], [delta_H_specific], color='blue') #spec

plt.xlabel('Temperature (K)')
plt.ylabel('ΔH (J/mol)')
plt.title('Enthalpy change ΔH(T)')
plt.legend()
plt.text(target_T, delta_H_specific, f'ΔH({target_T} K) = {delta_H_specific - interp_H(77):.2f}', # here we add the "extra" negative value from the extrapolated lower end
         verticalalignment='bottom', horizontalalignment='right')

#dS(T) plott
plt.subplot(1, 2, 2)
plt.plot(T_extended, Delta_S_extended, linestyle='-', color='r', label='ΔS(T)')
plt.scatter(T, Delta_S, color='red', label='Data Point')

plt.axvline(target_T, color='gray', linestyle='--') #spec
plt.axhline(delta_S_specific, color='gray', linestyle='--') #spec
plt.scatter([target_T], [delta_S_specific], color='red') #spec

plt.xlabel('Temperature (K)')
plt.ylabel('ΔS (J/K*mol)')
plt.title('Entropy change ΔS(T)')
plt.legend()
plt.text(target_T, delta_S_specific, f'ΔS({target_T} K) = {delta_S_specific - interp_S(77):.4f}', # here we add the "extra" negative value from the extrapolated lower end
         verticalalignment='bottom', horizontalalignment='right')

plt.tight_layout()
plt.show()
