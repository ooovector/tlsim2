import numpy as np


class QVariable:
    """
    Represents a variable of the circuit wavefunction or a constant external bias flux or charge.
    """

    def __init__(self, name):
        self.name = name
        self.variable_type = None
        self.variable_active = True
        self.phase_grid = None
        self.charge_grid = None
        self.phase_step = None
        self.charge_step = None
        self.nodeNo = None

    def create_grid(self, nodeNo, phase_periods, centre=0, centre_charge=0):
        """
        Creates a discrete grid for wavefunction variables.
        :param nodeNo: number of discrete points on the grid
        :param phase_periods: number of 2pi intervals in the grid
        :param centre: center of phase grid
        :param centre_charge: centre of charge grid (offset charge equivalent)
        """
        self.variable_type = 'variable'
        minNode = np.round(-nodeNo / 2)
        maxNode = minNode + nodeNo
        self.phase_grid = np.linspace(-np.pi * phase_periods + centre, np.pi * phase_periods + centre, nodeNo,
                                      endpoint=False)
        self.charge_grid = np.linspace(minNode / phase_periods + centre_charge, maxNode / phase_periods + centre_charge,
                                       nodeNo, endpoint=False)
        self.phase_step = 2 * np.pi * phase_periods / nodeNo
        self.charge_step = 1.0 / phase_periods
        self.nodeNo = nodeNo

    def set_parameter(self, phase_value, voltage_value):
        """
        Sets an external flux and/or charge bias.
        :param phase_value: external flux bias in flux quanta/(2pi)
        :param voltage_value: external charge bias in cooper pairs
        """
        self.variable_type = 'parameter'
        self.phase_grid = np.asarray([phase_value])
        self.charge_grid = np.asarray([voltage_value])
        self.phase_step = np.inf
        self.charge_step = np.inf
        self.nodeNo = 1

    def get_phase_grid(self):
        if self.variable_type == 'variable' and self.variable_active:
            return self.phase_grid
        elif self.variable_type == 'variable' and not self.variable_active:
            return np.zeros(1)
        else:
            return self.phase_grid

    def get_charge_grid(self):
        return self.charge_grid

    def get_phase_step(self):
        return self.phase_step

    def get_charge_step(self):
        return self.charge_step

    def get_nodeNo(self):
        return self.nodeNo
