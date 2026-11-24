import math
import pytest

from launchsim.constants import G0, MU_EARTH, R_EARTH, P0
from launchsim.staging import Stage, Vehicle, best_split, two_stage_delta_v, delta_v

def test_mass_ratio_e_gives_isp_times_g0():
    assert delta_v(300, math.e * 1000, 1000) == pytest.approx(300 * G0)

def test_bad_masses_raise():
    with pytest.raises(ValueError):
        delta_v(300, 1000, 2000)

def test_vehicle_total_is_sum_of_stages():
    s1 = Stage("S1", dry_mass=2000, propellant_mass=18000, isp_vac=300, thrust_vac=400000)
    s2 = Stage("S2", dry_mass=500, propellant_mass=4500, isp_vac=340, thrust_vac=50000)
    rocket = Vehicle([s1, s2], payload_mass=500)
    assert rocket.wet_mass == 25500
    dvs = rocket.stage_delta_vs()
    assert len(dvs) == 2
    assert rocket.total_delta_v() == pytest.approx(sum(dvs))
    # Stage 1 starts at the full 25 500 kg and ends at 25 500 - 18 000 = 7 500 kg
    assert dvs[0] == pytest.approx(300 * G0 * math.log(25500 / 7500))


def test_best_split_beats_equal_split():
    total, payload, epsilon, isp = 100_000.0, 1_000.0, 0.10, 320.0
    m1, dv_best = best_split(total, payload, epsilon, epsilon, isp, isp)
    dv_half = two_stage_delta_v((total - payload) / 2, total, payload, epsilon, epsilon, isp, isp)
    assert dv_best > dv_half


def test_identical_stages_have_equal_payload_ratios_at_optimum():
    # Classic result: with equal Isp and structural fraction, the optimal split
    # gives every stage the same "payload ratio" (mass above / mass at ignition).
    total, payload, epsilon, isp = 100_000.0, 1_000.0, 0.10, 320.0
    m1, _ = best_split(total, payload, epsilon, epsilon, isp, isp)
    m_initial_2 = total - m1
    ratio_1 = m_initial_2 / total
    ratio_2 = payload / m_initial_2
    assert ratio_1 == pytest.approx(ratio_2, rel=1e-3)