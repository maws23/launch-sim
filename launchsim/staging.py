import math
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from .constants import G0, MU_EARTH, R_EARTH, P0

def delta_v(isp, m_initial, m_final):
    if m_final <= 0 or m_initial <= m_final:
        raise ValueError("required that m_initial > m_final > 0")
    return isp * G0 * math.log(m_initial / m_final) # tsiolkovsky rocket equation

@dataclass
class Stage:
    name: str
    dry_mass: float #kg
    propellant_mass: float #kg
    isp_vac: float #specific impulse vacuum, seconds
    thrust_vac: float #thrust in vacuum, N

    @property
    def wet_mass(self):
        return self.dry_mass + self.propellant_mass

    @property
    def mass_flow(self):
        return self.thrust_vac / (self.isp_vac * G0) #propellant burn rate, kg/s

    @property
    def burn_time(self):
        return self.propellant_mass / self.mass_flow

class Vehicle:
    def __init__(self, stages, payload_mass):
        self.stages = list(stages)
        self.payload_mass = payload_mass

    def mass_above(self, i):
        return sum(s.wet_mass for s in self.stages[i + 1:]) + self.payload_mass #cumulative mass of stages above "i"th stage

    @property
    def wet_mass(self):
        return sum(s.wet_mass for s in self.stages) + self.payload_mass

    def stage_delta_vs(self):
        result = []
        for i, stage in enumerate(self.stages):
            m_initial = stage.wet_mass + self.mass_above(i)
            m_final = stage.dry_mass + self.mass_above(i)
            result.append(delta_v(stage.isp_vac, m_initial, m_final))
        return result

    def total_delta_v(self):
        return sum(self.stage_delta_vs())

def two_stage_delta_v(m1_wet, total_mass, payload, eps1, eps2, isp1, isp2):
    #Total ideal delta-v for a two-stage rocket, given stage 1's wet mass. eps = epsilon
    m2_wet = total_mass - payload - m1_wet
    m_initial_1 = total_mass
    m_final_1 = total_mass - (1 - eps1) * m1_wet
    m_initial_2 = m2_wet + payload
    m_final_2 = m_initial_2 - (1 - eps2) * m2_wet
    return delta_v(isp1, m_initial_1, m_final_1) + delta_v(isp2, m_initial_2, m_final_2)


def best_split(total_mass, payload, eps1, eps2, isp1, isp2):
    #Find the stage 1 wet mass that maximises total delta-v. Returns (m1_wet, delta_v)
    def negative_dv(m1_wet):
        return -two_stage_delta_v(m1_wet, total_mass, payload, eps1, eps2, isp1, isp2)

    result = minimize_scalar(negative_dv, bounds=(1.0, total_mass - payload - 1.0),
                             method="bounded", options={"xatol": 1e-6})
    return result.x, -result.fun