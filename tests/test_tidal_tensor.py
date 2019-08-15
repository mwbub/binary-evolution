import unittest
import numpy as np
from galpy.potential import KeplerPotential, TwoPowerTriaxialPotential

# noinspection PyProtectedMember
from binary_evolution.tidal_tensor import TidalTensor, _ttensor_galpy


class TidalTensorUnitTests(unittest.TestCase):

    # noinspection PyTypeChecker
    def setUp(self):
        """
        Set up TidalTensor instances for these unittests.
        """
        self.kepler_pot = KeplerPotential()
        self.kepler_tt = TidalTensor(self.kepler_pot)

        self.axisymmetric_pot = TwoPowerTriaxialPotential(c=0.7)
        self.axisymmetric_tt = TidalTensor(self.axisymmetric_pot)

        self.triaxial_pot = TwoPowerTriaxialPotential(c=0.7, b=0.95)
        self.triaxial_tt = TidalTensor(self.triaxial_pot)

        self.kepler_axi_pot = [self.kepler_pot, self.axisymmetric_pot]
        self.kepler_axi_tt = TidalTensor(self.kepler_axi_pot)

        self.kepler_triaxi_pot = [self.kepler_pot, self.triaxial_pot]
        self.kepler_triaxi_tt = TidalTensor(self.kepler_triaxi_pot)

    def tearDown(self):
        """
        Delete the attributes created in setUp.
        """
        del self.kepler_triaxi_tt
        del self.kepler_triaxi_pot

        del self.kepler_axi_tt
        del self.kepler_axi_pot

        del self.triaxial_tt
        del self.triaxial_pot

        del self.axisymmetric_tt
        del self.axisymmetric_pot

        del self.kepler_tt
        del self.kepler_pot

    def test_galpy_equivalence(self):
        """
        Test that TidalTensor behaves equivalently to galpy.potential.ttensor.
        """
        R = 1
        z = 1
        phi = np.pi / 2
        x = R * np.cos(phi)
        y = R * np.sin(phi)

        galpy_tt = _ttensor_galpy(self.kepler_axi_pot, x, y, z, 0)
        tt = self.kepler_axi_tt(x, y, z)

        self.assertTrue(np.all(np.isclose(tt, galpy_tt)),
                        msg="TidalTensor disagrees with galpy's ttensor")

    def test_summation(self):
        """
        Test that the TidalTensor is summed correctly.
        """
        x = 1
        y = 1
        z = 1

        tt1 = self.kepler_tt(x, y, z)
        tt2 = self.axisymmetric_tt(x, y, z)
        tt3 = self.kepler_axi_tt(x, y, z)

        self.assertTrue(np.all(tt1 + tt2 == tt3),
                        msg="TidalTensor does not sum correctly")

        tt4 = self.triaxial_tt(x, y, z)
        tt5 = self.kepler_triaxi_tt(x, y, z)

        self.assertTrue(np.all(tt1 + tt4 == tt5),
                        msg="TidalTensor does not sum correctly")

    def test_2ndderiv_hack(self):
        """
        Test that the second derivative hack is working correctly.
        """
        x = 1
        y = 1
        z = 1

        tt = self.axisymmetric_tt(x, y, z)
        self.assertEqual(tt[1, 1], tt[0, 0],
                         msg="Second derivative calculated "
                             "incorrectly for axisymmetric potentials")

        self.assertIsNot(self.axisymmetric_tt._tts[0], _ttensor_galpy,
                         msg="Second derivative hack not being used for an "
                             "axisymmetric potential")

    def test_triaxial_pot(self):
        """
        Test that TidalTensor works with triaxial potentials.
        """
        x = 1
        y = 1
        z = 1

        tt = self.triaxial_tt(x, y, z)
        self.assertGreater(tt[1, 1], tt[0, 0],
                           msg="TidalTensor fails in triaxial potentials")
