# ~~~~~~~~~~~~ Author: Mengmeng Cai ~~~~~~~~~~~~~~~

import opendssdirect as dss
import os
from scipy.sparse import csc_matrix
import numpy as np
import pandas as pd


class DSS(object):

    def __init__(self, dict_DSS, MainDir):
        """ Function used for initializing the DSS object

        Args:
            dict_DSS: A dictionary with five keys 'TIME', 'feedername', 'masterfile', 'loadshapefile and 'PVshapefile', dict
                -'TIME': Time stamp for simulation, str
                -'feedername': Folder name of the OpenDSS model, str
                -'masterfile': Name of the masterfile, str
                -'loadshapefile': Name of the loadshape file, str
                -'PVshapefile': Name of the PV generation shape file, str
            MainDir: Main directory of the DLMP project
        """

        self.MainDir = MainDir
        self.FeederDir = os.path.join(self.MainDir, dict_DSS['feedername'])
        self.MasterFile = os.path.join(self.FeederDir, dict_DSS['masterfile'])
        self.LoadShapeFile = os.path.join(self.FeederDir, dict_DSS['loadshapefile'])
        self.dssObj = dss
        self.dssObj.run_command('Compile ' + self.MasterFile)
        self.dssText = self.dssObj.Text
        self.dssCircuit = self.dssObj.Circuit
        self.dssSolution = self.dssObj.Solution
        self.dssLoad = self.dssObj.Loads
        self.dssPV = self.dssObj.PVsystems
        self.dssGenerator = self.dssObj.Generators
        self.dssVsource = self.dssObj.Vsources
        self.dsscktElement = self.dssObj.CktElement
        self.Sbus = dict_DSS['Sbus']
        self.ThreePhaseBus = dict_DSS['ThreePhaseBus']
        self.BaseS = dict_DSS['BaseS']
        self.BaseV = dict_DSS['BaseV']
        self.BaseY = dict_DSS['BaseY']

    def daily_run_config(self):
        """ Function used for configuring the daily mode simulation run
        """

        self.dssObj.run_command('Batchedit Load..* enabled=yes')
        self.dssObj.run_command('Batchedit Vsource..* enabled=yes')
        # self.dssObj.run_command('Batchedit Generator..* enabled=yes')
        self.dssObj.run_command('Redirect ' + self.LoadShapeFile)
        # self.dssObj.run_command('Redirect ' + self.PVShapeFile)
        self.dssObj.run_command('Set mode=daily number=1 stepsize=1h')

    def solve_singlestep(self):
        self.dssSolution.Solve()

    def getYmatrix(self):
        """ Function used for extracting system Y matrix from OpenDSS model

        Return:
            Y: System Y matrix, in pu
        """

        self.dssObj.run_command('Batchedit Load..* enabled=no')
        self.dssObj.run_command('Batchedit Vsource..* enabled=no')
        self.dssObj.run_command('Batchedit Generator..* enabled=no')
        self.dssObj.run_command('solve')
        Ysparse = csc_matrix(self.dssObj.YMatrix.getYsparse())
        Y = np.multiply(Ysparse.toarray(), self.BaseY)

        self.Y00 = Y[:3, :3]
        self.Y0L = Y[:3, 3:]
        self.YL0 = Y[3:, :3]
        self.YLL = Y[3:, 3:]

        return Y

    def getVol_daily(self):
        V_real_ts = []
        V_imag_ts = []
        V_ts = []
        for i in range(24):
            self.solve_singlestep()
            V_real, V_imag, V = self.getVol()
            V_real_ts.append(V_real)
            V_imag_ts.append(V_imag)
            V_ts.append(V)
        V_real_ts = np.array(V_real_ts).T
        V_imag_ts = np.array(V_imag_ts).T
        V_ts = np.array(V_ts).T
        return V_real_ts, V_imag_ts, V_ts

    def getVol(self):
        """ Function used for extracting Voltage phasor information at all nodes from OpenDSS model

        Return:
            V: Complex numpy array of voltage phasors at all nodes, in pu
        """

        Vol = [v / self.BaseV for v in self.dssCircuit.YNodeVArray()]
        V_real = np.array([Vol[i] for i in range(len(Vol)) if i % 2 == 0])
        V_imag = np.array([Vol[i] for i in range(len(Vol)) if i % 2 == 1])
        V = V_real + np.multiply(1j, V_imag)
        return V_real, V_imag, V

    def getSubstationInj(self):
        """ Function used for extracting power injection information at only slack bus from OpenDSS model

        Return:
            Sinj: Complex numpy array of power injections at only slack bus, in pu
        """

        Nodes = self.getNodes()
        Sinj = pd.DataFrame(np.complex(0), index=Nodes, columns=['Sinj'])
        for Source in self.dssVsource.AllNames():
            self.dssCircuit.SetActiveElement('Vsource.' + Source)
            Sinj.Sinj[self.Sbus+'.1'] = -(
                    self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) / self.BaseS
            Sinj.Sinj[self.Sbus+'.2'] = -(
                    self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])) / self.BaseS
            Sinj.Sinj[self.Sbus+'.3'] = -(
                    self.dsscktElement.Powers()[4] + np.multiply(1j, self.dsscktElement.Powers()[5])) / self.BaseS
        return Sinj.values.real, Sinj.values.imag, Sinj.values

    def getSubstationInj_daily(self):
        P0_ts = []
        Q0_ts = []
        S0_ts = []
        for i in range(24):
            self.solve_singlestep()
            P0, Q0, S0 = self.getSubstationInj()
            P0_ts.append(P0)
            Q0_ts.append(Q0)
            S0_ts.append(S0)
        P0_ts = np.array(P0_ts).T
        Q0_ts = np.array(Q0_ts).T
        S0_ts = np.array(S0_ts).T
        return P0_ts, Q0_ts, S0_ts

    def getSinj(self):
        """ Function used for extracting power injection information at only slack bus from OpenDSS model

        Return:
            Sinj: Complex numpy array of power injections at only slack bus, in pu
        """

        Nodes = self.getNodes()
        Sinj = pd.DataFrame(np.complex(0), index=Nodes, columns=['Sinj'])
        for Source in self.dssVsource.AllNames():
            self.dssCircuit.SetActiveElement('Vsource.' + Source)
            Sinj.Sinj['800.1'] = -(
                    self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) / self.BaseS
            Sinj.Sinj['800.2'] = -(
                    self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])) / self.BaseS
            Sinj.Sinj['800.3'] = -(
                    self.dsscktElement.Powers()[4] + np.multiply(1j, self.dsscktElement.Powers()[5])) / self.BaseS

        for Load in self.dssLoad.AllNames():
            self.dssCircuit.SetActiveElement('Load.' + Load)
            if Load[-1] == 'a':
                Sinj.Sinj[Load[1:-1] + '.1'] = -(
                        self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) / self.BaseS
            elif Load[-1] == 'b':
                Sinj.Sinj[Load[1:-1] + '.2'] = -(
                        self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) / self.BaseS
            elif Load[-1] == 'c':
                Sinj.Sinj[Load[1:-1] + '.3'] = -(
                        self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) / self.BaseS
        return Sinj.values.real, Sinj.values.imag, Sinj.values

    def getSinj_daily(self):
        P_ts = []
        Q_ts = []
        S_ts = []
        for i in range(24):
            self.solve_singlestep()
            P, Q, S = self.getSinj()
            P_ts.append(P)
            Q_ts.append(Q)
            S_ts.append(S)
        P_ts = np.array(P_ts).T
        Q_ts = np.array(Q_ts).T
        S_ts = np.array(S_ts).T
        return P_ts, Q_ts, S_ts

    def getNodes(self):
        """ Function used for extracting nodes name from OpenDSS model

        Return:
            Nodes: list of node names
        """

        Nodes = np.array(self.dssCircuit.YNodeOrder())
        return Nodes

if __name__ == "__main__":

    dict_DSS = {
        'Sbus': '800',
        'ThreePhaseBus': [],
        'BaseS': 1,
        'BaseV': 2401.77711983,
        'BaseY': 5768.533333333333582,
        'feedername': 'IEEE34Bus',
        'masterfile': 'ieee34Mod1.dss',
        'loadshapefile': 'SetDailyLoadShape.DSS'
    }

    IEEE34 = DSS(dict_DSS, os.getcwd())

    Y = IEEE34.getYmatrix()
    Nodes = IEEE34.getNodes()

    IEEE34.daily_run_config()

    Vreal_24, Vimag_24, V_24 = IEEE34.getVol_daily()
    IEEE34.daily_run_config()
    P_24, Q_24, S_24 = IEEE34.getSinj_daily()
    pd.DataFrame(Y.real, index=Nodes, columns=Nodes).to_csv(IEEE34.MainDir+'/Data/Yreal.csv')
    pd.DataFrame(Y.imag, index=Nodes, columns=Nodes).to_csv(IEEE34.MainDir+'/Data/Yimag.csv')
    pd.DataFrame(V_24.real, index=Nodes, columns=range(24)).to_csv(IEEE34.MainDir+'/Data/Vreal.csv')
    pd.DataFrame(V_24.imag, index=Nodes, columns=range(24)).to_csv(IEEE34.MainDir+'/Data/Vimag.csv')
    pd.DataFrame(np.absolute(V_24), index=Nodes, columns=range(24)).to_csv(IEEE34.MainDir + '/Data/Vmag.csv')
    pd.DataFrame(P_24[0, :, :], index=Nodes, columns=range(24)).to_csv(IEEE34.MainDir+'/Data/P.csv')
    pd.DataFrame(Q_24[0, :, :], index=Nodes, columns=range(24)).to_csv(IEEE34.MainDir+'/Data/Q.csv')



    mismatch = []
    for i in range(24):
        diff = np.add(S_24[0, :, i:i+1], - np.multiply(V_24[:, i:i+1], np.conjugate(np.matmul(Y, V_24[:, i:i+1]))))
        mismatch.append(diff)
    print(np.array(mismatch).T[0])
    np.savetxt(IEEE34.MainDir+'/Data/mismatch.csv', np.absolute(np.array(mismatch).T[0]), delimiter=',')


    # np.savetxt(IEEE34.MainDir+'/Data/Init/P0.csv', P0_24[0, :3, :].T, delimiter=',')
    # np.savetxt(IEEE34.MainDir+'/Data/Init/Q0.csv', Q0_24[0, :3, :].T, delimiter=',')
    # np.savetxt(IEEE34.MainDir+'/Data/Init/Vreal.csv', V_24.real[3:, :].T, delimiter=',')
    # np.savetxt(IEEE34.MainDir+'/Data/Init/Vimag.csv', V_24.imag[3:, :].T, delimiter=',')
    # np.savetxt(IEEE34.MainDir+'/Data/Init/Vmag.csv', np.absolute(V_24)[3:, :].T, delimiter=',')



