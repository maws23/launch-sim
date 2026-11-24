import matplotlib.pyplot as plt
import numpy as np

from launchsim.staging import Stage, Vehicle, best_split, two_stage_delta_v

s1 = Stage("Stage 1", dry_mass=2000, propellant_mass=18000, isp_vac=300, thrust_vac=400000)
s2 = Stage("Stage 2", dry_mass=500, propellant_mass=4500, isp_vac=340, thrust_vac=50000)
rocket = Vehicle([s1, s2], payload_mass=500)
for stage, dv in zip(rocket.stages, rocket.stage_delta_vs()):
    print(f"{stage.name} delta-v: {dv:.0f} m/s")
print(f"Total ideal delta_v: {rocket.total_delta_v():.0f} m/s\n")
#ideal delta-v of 2 stage rocket

total, payload, epsilon, isp = 100000, 1000, 0.1, 320
m1_best, dv_best = best_split(total, payload, epsilon, epsilon, isp, isp) #epsilon = dry mass / total mass
print(f"Best split: Stage 1 wet mass {m1_best:.0f} kg, Stage 2 wet mass "
      f"{total - payload - m1_best:.0f} kg, delta-v {dv_best:.0f} m/s")
#splitting a given launch mass into 2 stages to maximise total ideal delta-v

m1_values = np.linspace(5000, total - payload - 5000, 300)
dv_values = [two_stage_delta_v(m1, total, payload, epsilon, epsilon, isp, isp) for m1 in m1_values]

plt.plot(m1_values / 1000, np.array(dv_values) / 1000)
plt.plot(m1_best / 1000, dv_best / 1000, "ro", label="optimum")
plt.xlabel("Stage 1 wet mass (tonnes)")
plt.ylabel("Total ideal delta-v (km/s)")
plt.title("Splitting 100 t between two stages")
plt.legend()
plt.grid(True)
plt.show()
