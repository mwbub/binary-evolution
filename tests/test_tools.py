import warnings
import unittest
import numpy as np
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, evaluatePotentials
from galpy.util.bovy_conversion import time_in_Gyr

from binary_evolution.tools import v_circ, v_esc, period, ecc_to_vel

# Factors for conversion from galpy internal units
_pc = 8000
_kms = 220
_yr = time_in_Gyr(220, 8) * 1e+9


class ToolsUnitTests(unittest.TestCase):

    def test_v_circ(self):
        R = z = phi = 1
        vc = v_circ(MWPotential2014, [R, z, phi])
        orb = Orbit(vxvv=[R/_pc, 0, vc/_kms, z/_pc, 0, phi])
        ecc = orb.e(pot=MWPotential2014, analytic=True)

        self.assertLess(ecc, 0.01, msg="v_circ returns incorrect circular "
                                       "velocity")

    def test_v_esc(self):
        R = z = phi = 1
        ve = v_esc(MWPotential2014, [R, z, phi])
        orb = Orbit(vxvv=[R/_pc, 0, ve/_kms, z/_pc, 0, phi])
        E = orb.E(pot=MWPotential2014)
        E_inf = evaluatePotentials(MWPotential2014, 1e+12, 0)

        self.assertAlmostEqual(E, E_inf, 10, msg="v_esc returns incorrect "
                                                 "escape velocity")

    def test_period(self):
        R = z = phi = 1 * _pc
        v_R = v_z = v_phi = 1 * _kms
        P = period(MWPotential2014, [R, z, phi], [v_R, v_z, v_phi])
        orb = Orbit(vxvv=[R/_pc, v_R/_kms, v_phi/_kms, z/_pc, v_z/_kms, phi])
        P_galpy = orb.Tp(pot=MWPotential2014, analytic=True) * _yr
        diff = np.abs(2 * (P - P_galpy) / (P + P_galpy))

        self.assertLess(diff, 0.1, msg="period returns incorrect period")

    def test_ecc_to_vel(self):
        R = z = phi = 1

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v0 = ecc_to_vel(MWPotential2014, 0, [R, z, phi])
            v05 = ecc_to_vel(MWPotential2014, 0.5, [R, z, phi])

        orb0 = Orbit(vxvv=[R/_pc, 0, v0/_kms, z/_pc, 0, phi])
        orb05 = Orbit(vxvv=[R/_pc, 0, v05/_kms, z/_pc, 0, phi])

        e0 = orb0.e(pot=MWPotential2014, analytic=True)
        e05 = orb05.e(pot=MWPotential2014, analytic=True)

        self.assertLess(e0, 0.01, msg="ecc_to_vel returns incorrect circular "
                                      "velocity")
        self.assertAlmostEqual(e05, 0.5, 2, msg="ecc_to_vel returns incorrect "
                                                "eccentric velocity")
