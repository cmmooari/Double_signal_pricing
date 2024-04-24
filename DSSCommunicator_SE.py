# ~~~~~~~~~~~~ Author: Mengmeng Cai ~~~~~~~~~~~~~~~
import opendssdirect as dss
import os
from scipy.sparse import csc_matrix
import numpy as np
import pandas as pd
from datetime import datetime
import time
import re

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
        self.Sbus = dict_DSS['Sbus']
        self.Sbase = dict_DSS['Sbase']

        # Compiling the OpenDSS file
        start = time.time()
        self.dssObj = dss
        self.dssObj.run_command('Compile ' + self.MasterFile)
        self.dssObj.run_command('get datapath')
        print(self.dssObj.Text.Result())
        print("Time used for compiling the opendss file", time.time()  -start)

        # Claiming useful OpenDSS objects
        self.dssText  = self.dssObj.Text
        self.dssCircuit = self.dssObj.Circuit
        self.dssSolution = self.dssObj.Solution
        self.dssLine = self.dssObj.Lines
        self.dssTransformer = self.dssObj.Transformers
        self.dssLoad  = self.dssObj.Loads
        self.dssPV = self.dssObj.PVsystems
        self.dssGenerator =  self.dssObj.Generators
        self.dssVsource  = self.dssObj.Vsource
        self.dssMonitor = self.dssObj.Monitors
        self.dsscktElement = self.dssObj.CktElement
        self.dssBus = self.dssObj.Bus

    def static_run_config(self):
        """ Function used for configuring the steady-state simulation
        """

        self.dssObj.run_command('Batchedit Load..* enabled=yes')
        self.dssObj.run_command('Batchedit Vsource..* enabled=yes')
        self.dssObj.run_command('Batchedit Generator..* enabled=yes')
        self.dssObj.run_command('Batchedit Line..* enabled=yes')
        self.dssObj.run_command('Batchedit Transformer..* enabled=yes')
        self.dssObj.run_command('Batchedit CapControl..* enabled=no')
        self.dssObj.run_command('Batchedit RegControl..* enabled=no')
        self.dssObj.run_command('Batchedit Fuse..* enabled=no')
        # self.dssObj.run_command('Batchedit Transformer..* wdg=2')
        # self.dssObj.run_command('Batchedit Transformer..* tap=1')
        self.dssObj.run_command('Set mode=daily stepsize=1m')

    def dynamic_run_config(self):
        """ Function used for configuring the dynamic simulation
        """

        self.dssObj.run_command('Batchedit Load..* enabled=yes')
        self.dssObj.run_command('Batchedit Vsource..* enabled=yes')
        self.dssObj.run_command('Batchedit Generator..* enabled=yes')
        self.dssObj.run_command('Batchedit Line..* enabled=yes')
        self.dssObj.run_command('Batchedit Transformer..* enabled=yes')
        self.dssObj.run_command('Batchedit CapControl..* enabled=no')
        self.dssObj.run_command('Batchedit RegControl..* enabled=no')
        self.dssObj.run_command('Set mode=daily stepsize=5s')

    def duration_run(self, period):
        """ Function used for running the time-series simulation for a given period
        """

        self.dssObj.run_command("Set Loadshapeclass=daily")
        self.dssObj.run_command("solve number="+str(period))

    def snapshot_run(self):
        """ Function used for running the snapshot simulation
        """

        self.dssObj.run_command("Set Loadshapeclass=daily")
        self.dssObj.run_command("solve number=1")

    def time_indexed_run(self, TIME):
        """ Function used for running the snapshot simulation at pre-defined time step
        """

        self.dssObj.run_command("Set Loadshapeclass=daily")
        self.dssSolution.Hour(TIME.hour)
        self.dssSolution.Seconds(TIME.minute * 60 + TIME.second)
        self.dssObj.run_command('solve')

    def getYmatrix(self):
        """ Function used for extracting system Y matrix from OpenDSS model

        Return:
            Y: sparse system Y matrix, in absolute value
            Ynode: node indexs of Y
        """

        self.dssObj.run_command('Batchedit Load..* enabled=no')
        self.dssObj.run_command('Batchedit Vsource..* enabled=no')
        self.dssObj.run_command('Batchedit Generator..* enabled=no')
        self.dssObj.run_command('Batchedit Line..* enabled=yes')
        self.dssObj.run_command('Batchedit Transformer..* enabled=yes')
        self.dssObj.run_command('Batchedit CapControl..* enabled=no')
        self.dssObj.run_command('Batchedit RegControl..* enabled=no')
        self.dssObj.run_command('Batchedit Fuse..* enabled=no')
        # self.dssObj.run_command('Batchedit Transformer..* wdg=2')
        # self.dssObj.run_command('Batchedit Transformer..* tap=1')
        self.dssObj.run_command('solve')
        Ynodes = self.dssCircuit.YNodeOrder()
        Y = csc_matrix(self.dssObj.Ymatrix.getYspare(factor=False))

        return Y, Ynodes

    def getYmatrix_disTrans(self, secondary_transformers):
        """ Function used for extracting system Y matrix from OpenDSS model

        Return:
            Y: sparse system Y matrix, in absolute value
            Ynode: node indexs of Y
        """

        self.dssObj.run_command('Batchedit Load..* enabled=no')
        self.dssObj.run_command('Batchedit Vsource..* enabled=no')
        self.dssObj.run_command('Batchedit Generator..* enabled=no')
        self.dssObj.run_command('Batchedit Line..* enabled=yes')
        self.dssObj.run_command('Batchedit CapControl..* enabled=no')
        self.dssObj.run_command('Batchedit RegControl..* enabled=no')
        self.dssObj.run_command('Batchedit Fuse..* enabled=no')
        for transformer in secondary_transformers:
            self.dssObj.run_command('Transformer.' + transformer + '.enabled=no')
        self.dssObj.run_command('solve')
        Ynodes = self.dssCircuit.YNodeOrder()
        Y = csc_matrix(self.dssObj.Ymatrix.getYspare(factor=False))

        return Y, Ynodes
    
    
    def getYmatrix_disLines(self, secondary_lines):
        """ Function used for extracting system Y matrix from OpenDSS model

        Return:
            Y: sparse system Y matrix, in absolute value
            Ynode: node indexs of Y
        """

        self.dssObj.run_command('Batchedit Load..* enabled=no')
        self.dssObj.run_command('Batchedit Vsource..* enabled=no')
        self.dssObj.run_command('Batchedit Generator..* enabled=no')
        self.dssObj.run_command('Batchedit Transformer..* enabled=yes')
        self.dssObj.run_command('Batchedit CapControl..* enabled=no')
        self.dssObj.run_command('Batchedit RegControl..* enabled=no')
        self.dssObj.run_command('Batchedit Fuse..* enabled=no')
        for line in secondary_lines:
            self.dssObj.run_command('Line.' + line + '.enabled=no')
        self.dssObj.run_command('solve')
        Ynodes = self.dssCircuit.YNodeOrder()
        Y = csc_matrix(self.dssObj.Ymatrix.getYspare(factor=False))

        return Y, Ynodes


    def getVmag(self):
        """ Function used for getting the voltage magnitudes at each node 

        Returns:
            Vmag: DataFrame of voltage magnitude indexed by nodes, in V
        """
        
        Vol = self.dssCircuit.YNodeVArray()
        V_real = np.array([Vol[i] for i in range(len(Vol)) if i % 2 == 0])
        V_imag = np.array([Vol[i] for i in range(len(Vol)) if i % 2 == 1])
        V = V_real + np.multiply(1j, V_imag)
        Vnodes = self.getNodes()
        Vmag = pd.DataFrame(np.absolute(V), index=Vnodes, columns=["Vmag"])
        return Vmag

    def getVangle(self):
        """ Function used for extracting Voltage angle information at all nodes from OpenDSS model

        Return:
            Vangle: Dataframe of voltage angles indexed by nodes, in degree
        """

        Vol = self.dssCircuit.YNodeVArray()
        V_real = np.array([Vol[i] for i in range(len(Vol)) if i % 2 == 0])
        V_imag = np.array([Vol[i] for i in range(len(Vol)) if i % 2 == 1])
        V = V_real + np.multiply(1j, V_imag)
        Vnodes = self.getNodes()
        Vangle = pd.DataFrame(np.angle(V), index=Vnodes, columns=["Vangle"])
        return Vangle

    def getV(self):
        Vol = self.dssCircuit.YNodeVArray()
        V_real = np.array([Vol[i] for i in range(len(Vol)) if i % 2 == 0])
        V_imag = np.array([Vol[i] for i in range(len(Vol)) if i % 2 == 1])
        Vnodes = self.getNodes()
        V = pd.DataFrame(V_real + np.multiply(1j, V_imag), index=Vnodes, columns=["Vangle"])
        return V

    def getI(self):
        """ Function for extracting current magnitudes at each node
        
        Return:
            Vmag: numpy array of current magnitude at all nodes, in A
        """

        I = self.dssCircuit.YCurrents()
        I_real = np.array([I[i] for i in range(len(I)) if i % 2 == 0])
        I_imag = np.array([I[i] for i in range(len(I)) if i % 2 == 1])
        I = I_real + np.multiply(1j, I_imag)
        Inodes = self.getNodes()
        Ipd = pd.DataFrame(I, index=Inodes, columns=["I"])
        return Ipd

    def getVbase(self):
        """ Function for extracting voltage base at each node
        
        Return:
            Vbase: Dataframe of base voltage values indexed by nodes, in V
        """

        Vbase = pd.DataFrame(np.zeros, index=self.getNodes(), columns=['Vbase'])
        for node in self.getNodes():
            self.dssCircuit.SetActiveBus(node)
            Vbase.Vbase[node] = self.dssBus.kVBase() * 1000
        return Vbase
    
    def getVoltagePUbyPbase(self, phase):
        """ Function for extracting voltage values by phases
        
        Return:
            Vbase: Dataframe of voltage magnitudes indexed by bus on a specific phase, in pu
        """
        Vphase = pd.DataFrame(np.array(self.dssCircuit.AllNodeVmagPUByPhase(phase)),
                              index=[bus[:-2] for bus in self.dssCircuit.AllNodeNamesByPhase(phase)],
                              columns=['Vmag'])
        return Vphase

    def addDG_single(self, name, node, kW, kVar):
        self.dssCircuit.SetActiveBus(node)
        kV = self.dssBus.kVBase()
        self.dssObj.run_command(
            'New Generator.' + name + 'Bus1=' + node + 'Phase=1 Conn=Wye Model=1 kV=' + str(kV) + \
                 'kW=' + str(kW) + 'kvar=' + str(kVar) + 'Vminpu=0.80 Vmaxpu=1.20'
        )

    def addDG_three(self, name, bus, kW, kVar):
        self.dssCircuit.SetActiveBus(bus)
        kV = self.dssBus.kVBase()
        self.dssObj.run_command(
            'New Generator.' + name + 'Bus1=' + bus + 'Phase=3 Conn=Wye Model=1 kV=' + str(kV) + \
                 'kW=' + str(kW) + 'kvar=' + str(kVar) + 'Vminpu=0.80 Vmaxpu=1.20'
        )

    def dispatchDG(self, name, kW=None, kVar=None):
        self.dssGenerator.Name(name)
        
        if kW != None:
            self.dssGenerator.kW(kW)
        
        if kVar != None:
            self.dssGenerator.kvar(kVar)

    def getS(self):
        Nodes = self.getNodes()
        Sinj = pd.DataFrame(np.complex(0), index=Nodes, columns=["S"])
        for Source in self.dssVsource.AllNames():
            self.dssCircuit.SetActiveElement('Vsource.' + Source)
            Sinj.S[self.Sbus[0]] = self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])
            Sinj.S[self.Sbus[1]] = self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])
            Sinj.S[self.Sbus[2]] = self.dsscktElement.Powers()[4] + np.multiply(1j, self.dsscktElement.Powers()[5])

        for load in self.dssLoad.AllNames():
            self.dssCircuit.SetActiveElement('Load.' + load)
            for n in range(self.dsscktElement.NumPhases()):
                phase = self.dsscktElement.NodeOrder()[n]
                Sinj.S[self.dsscktElement.BusNames()[0][:-2 * self.dsscktElement.NumPhases()].upper() + '.' + str(phase)]\
                    += self.dsscktElement.Powers()[2*n] + np.multiply(1j, self.dsscktElement.Powers())[2*n+1]
        
        for DG in self.dssGenerator.AllNames():
            self.dssCircuit.SetActiveElement('Generator.' + DG)
            for n in range(self.dsscktElement.NumPhases()):
                phase = self.dsscktElement.NodeOrder()[n]
                Sinj.S[self.dsscktElement.BusNames()[0][:-2 * self.dsscktElement.NumPhases()].upper() + '.' + str(phase)]\
                    += self.dsscktElement.Powers()[2*n] + np.multiply(1j, self.dsscktElement.Powers())[2*n+1]
        return Sinj

    def getSubInj(self):
        TotalPower = self.dssCircuit.TotalPower()[0]/1000 + np.complex(1j) * self.dssCircuit.TotalPower()[1]/1000
        return TotalPower

    def getSubInjbyPhase(self):
        Slacknodes = self.Sbus
        TotalPower_byphase = pd.DataFrame(np.complex(0), index=Slacknodes, columns=['Sinj'])
        for Source in self.dssVsource.AllNames():
            self.dssCircuit.SetActiveElement('Vsource.' + Source)
            TotalPower_byphase.Sinj[self.Sbus[0]] = \
                self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])
            TotalPower_byphase.Sinj[self.Sbus[1]] = \
                self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])
            TotalPower_byphase.Sinj[self.Sbus[2]] = \
                self.dsscktElement.Powers()[4] + np.multiply(1j, self.dsscktElement.Powers()[5])
        return TotalPower_byphase

    def updateSubVmag(self, V):
        self.dssVsource.PU(V)

    def getLoads(self):
        Loads = pd.DataFrame(np.complex(0), index=self.getNodes(), columns=['load'])
        for load in self.dssLoad.AllNames():
            self.dssCircuit.SetActiveElement('Load.' + load)
            for n in range(self.dsscktElement.NumPhases()):
                phase = self.dsscktElement.NodeOrder()[n]
                Loads.load[self.dsscktElement.BusNames()[0][:-2*self.dsscktElement.NumPhases()].upper()+'.'+str(phase)]\
                    = self.dsscktElement.Powers()[2*n] + np.multiply(1j, self.dsscktElement.Powers()[2*n+1])
        return Loads
    
    def getLoadNodes(self):
        LoadNodes = [] 
        for load in self.dssLoad.AllNames(): 
            self.dssCircuit.SetActiveElement('Load.' + load) 
            for n in range(self.dsscktElement.NumPhases()): 
                phase = self.dsscktElement.NodeOrder()[n] 
                LoadNodes.append(self.dsscktElement.BusNames()[0][:-2*self.dsscktElement.NumPhases()].upper()+'.'+str(phase)) 
        return LoadNodes

    def getSwitchStatus(self): 
        Switches = self.getSwitches() 
        SwitchStatus_a = pd.DataFrame(np.nan, index=Switches, columns=['switch']) 
        SwitchStatus_b = pd.DataFrame(np.nan, index=Switches, columns=['switch']) 
        SwitchStatus_c = pd.DataFrame(np.nan, index=Switches, columns=['switch'])
       
        for switch in Switches: 
            self.dssCircuit.SetActiveElement('Line.'+switch) 
            if self.dsscktElement.Enabled() == True: 
                for n, phase in enumerate(self.dsscktElement.NodeOrder()[:self.dsscktElement.NumPhases()]): 
                    if phase == 1: 
                        SwitchStatus_a.switch[switch] = self.dsscktElement.IsOpen(2, n + 1) 
                    elif phase == 2: 
                        SwitchStatus_b.switch[switch] = self.dsscktElement.IsOpen(2, n + 1) 
                    else: 
                        SwitchStatus_c.switch[switch] = self.dsscktElement.IsOpen(2, n + 1)
        return SwitchStatus_a, SwitchStatus_b, SwitchStatus_c

    def getLineFlowbyPhase(self): 
    
        Lines = self.getLines() 
        LF_A_forward = pd.DataFrame(np.complex(0), index=Lines, columns=['Lineflow']) 
        LF_A_backward = pd.DataFrame(np.complex(0), index=Lines, columns=['Lineflow']) 
        LF_B_forward = pd.DataFrame(np.complex(0), index=Lines, columns=['Lineflow']) 
        LF_B_backward = pd.DataFrame(np.complex(0), index=Lines, columns=['Lineflow']) 
        LF_C_forward = pd.DataFrame(np.complex(0), index=Lines, columns=['Lineflow']) 
        LF_C_backward = pd.DataFrame(np.complex(0), index=Lines, columns=['Lineflow'])
        
        for Line in Lines: 
            self.dssCircuit.SetActiveElement('Line.' + Line)
            
            if self.dsscktElement.NumPhases() == 1: 
                if self.dsscktElement.NodeOrder()[0] == 1: 
                    LF_A_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) 
                    LF_A_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])) 
                elif self.dsscktElement.NodeOrder()[0] == 2: 
                    LF_B_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) 
                    LF_B_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])) 
                elif self.dsscktElement.NodeOrder()[0] == 3: 
                    LF_C_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) 
                    LF_C_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3]))
            if self.dsscktElement.NumPhases() == 2: 
                if self.dsscktElement.NodeOrder()[0] == 1: 
                    LF_A_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) 
                    LF_A_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[4] + np.multiply(1j, self.dsscktElement.Powers()[5])) 
                elif self.dsscktElement.NodeOrder()[0] == 2: 
                    LF_B_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) 
                    LF_B_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[4] + np.multiply(1j, self.dsscktElement.Powers()[5])) 
                elif self.dsscktElement.NodeOrder()[0] == 3: 
                    LF_C_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) 
                    LF_C_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[4] + np.multiply(1j, self.dsscktElement.Powers()[5]))
                if self.dsscktElement.NodeOrder()[2] == 1: 
                    LF_A_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])) 
                    LF_A_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[6] + np.multiply(1j, self.dsscktElement.Powers()[7])) 
                elif self.dsscktElement.NodeOrder()[2] == 2: 
                    LF_B_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])) 
                    LF_B_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[6] + np.multiply(1j, self.dsscktElement.Powers()[7])) 
                elif self.dsscktElement.NodeOrder()[2] == 3: 
                    LF_C_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])) 
                    LF_C_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[6] + np.multiply(1j, self.dsscktElement.Powers()[7]))
            if self.dsscktElement.NumPhases() >= 3: 
                LF_A_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[0] + np.multiply(1j, self.dsscktElement.Powers()[1])) 
                LF_B_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[2] + np.multiply(1j, self.dsscktElement.Powers()[3])) 
                LF_C_forward.Lineflow[Line] = -( self.dsscktElement.Powers()[4] + np.multiply(1j, self.dsscktElement.Powers()[5])) 
                LF_A_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[6] + np.multiply(1j, self.dsscktElement.Powers()[7])) 
                LF_B_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[8] + np.multiply(1j, self.dsscktElement.Powers()[9])) 
                LF_C_backward.Lineflow[Line] = -( self.dsscktElement.Powers()[10] + np.multiply(1j, self.dsscktElement.Powers()[11]))
        return LF_A_forward, LF_B_forward, LF_C_forward, LF_A_backward, LF_B_backward, LF_C_backward
 
    def getLineFlow(self):
        Lines = self.getSingleLines()
        LF_forward = pd.DataFrame(np.complex(0), index=Lines, columns=['Lineflow']) 
        LF_backward = pd.DataFrame(np.complex(0), index=Lines, columns=['Lineflow'])

        for line in self.getLines(): 
            self.dssCircuit.SetActiveElement('Line.' + line) 
            if self.dsscktElement.Enabled() == False: 
                pass 
            else: 
                NumPhases = self.dsscktElement.NumPhases() 
                for n in range(NumPhases): 
                    phase = self.dsscktElement.NodeOrder()[n] 
                    LF_forward.Lineflow[line + '.' + str(phase)] = self.dsscktElement.Powers()[2 * n] + np.multiply(1j, self.dsscktElement.Powers()[2 * n + 1]) 
                    LF_backward.Lineflow[line + '.' + str(phase)] - self.dsscktElement.Powers()[2 * n + 2 * NumPhases] \
                            - np.multiply(1j, self.dsscktElement.Powers()[2 * n + 1 + 2 * NumPhases])
        return LF_forward, LF_backward
    def getLineCurrent(self):
        Index = []
        for line in self.getLines(): 
            self.dssCircuit.SetActiveElement('Line.' + line) 
            if self.dsscktElement.Enabled() == False: 
                pass 
            else: 
                NumPhases = self.dsscktElement.NumPhases() 
                for n in range(NumPhases): 
                    phase = self.dsscktElement.NodeOrder()[n] 
                    sending_node = re.split("\.", self.dsscktElement.BusNames()[0])[0] 
                    receiving_node = re.split("\.", self.dsscktElement.BusNames()[1])[0] 
                    Index.append((sending_node + '.' + str(phase), receiving_node + '.' + str(phase))) 
                    Index.append((receiving_node + '.' + str(phase), sending_node + '.' + str(phase)))
        
        LC = pd.DataFrame(np.complex(0), index=Index, columns=['LineCurrent'])
        for line in self.getLines(): 
            self.dssCircuit.SetActiveElement('Line.' + line) 
            if self.dsscktElement.Enabled() == False: 
                pass 
            else: 
                NumPhases = self.dsscktElement.NumPhases() 
                for n in range(NumPhases): phase = self.dsscktElement.NodeOrder()[n] 
                sending_node = re.split("\.", self.dsscktElement.BusNames()[0])[0]
                receiving_node = re.split("\.", self.dsscktElement.BusNames()[1])[0] 
                LC.LineCurrent[(sending_node + '.' + str(phase), receiving_node + '.' + str(phase))] = \
                    self.dsscktElement.Currents()[2 * n] + np.multiply(1j, self.dsscktElement.Currents()[2 * n + 1])
                LC.LineCurrent[(receiving_node + '.' + str(phase), sending_node + '.' + str(phase))] = \
                    self.dsscktElement.Currents()[2 * n + 2 * NumPhases] + \
                    np.multiply(1j, self.dsscktElement.Currents()[2 * n + 1 + 2 * NumPhases])
        return LC
        
    def getNodes(self): 
        Nodes = np.array(self.dssCircuit.YNodeOrder()) 
        return Nodes
 
    def getLines(self): 
        Lines = np.array(self.dssLine.AllNames()) 
        return Lines
 
    def getTransformers(self): 
        Transformers = np.array(self.dssTransformer.AllNames()) 
        return Transformers
    
    def getSingleLines(self): 
        Lines = [] 
        for line in self.dssLine.AllNames(): 
            self.dssCircuit.SetActiveElement("Line."+line) 
            if self.dsscktElement.Enabled() == False: 
                pass 
            else: 
                for n in range(self.dsscktElement.NumPhases()): Lines.append(line+'.'+str(self.dsscktElement.NodeOrder()[n])) 
        return Lines
 
    def getBuses(self): 
        Buses = np.array(self.dssCircuit.AllBusNames()) 
        return Buses
 
    def getSwitches(self): 
        Switches = [] 
        for line in self.dssLine.AllNames(): 
            if "switch" in line: Switches.append(line) 
        return Switches

    def bus_coordinate(self): 
        buses = self.getBuses() 
        coordinate = pd.DataFrame(index=buses, columns=['x', 'y']) 
        for bus in buses: 
            self.dssCircuit.SetActiveBus(bus) 
            coordinate.loc[bus] = [self.dssBus.X(), self.dssBus.Y()] 
        return coordinate

    def bus_connection(self): 
        lines = self.getLines() 
        connection = pd.DataFrame(index=lines, columns=['Bus1', 'Bus2']) 
        for line in lines: 
            self.dssCircuit.SetActiveElement("line."+str(line)) 
            connection.loc[line] = [self.dsscktElement.BusNames()[0], self.dsscktElement.BusNames()[1]]
        return connection
            
        

if __name__ == "__main__":

    dict_DSS = {
        'Sbus': ['100.a', '100.2', '100.3'],
        'Sbase': 100,
        'feedername': 'IEEE34Bus',
        'masterfile': 'ieee34Mod1.dss',
        'TIME': '1:01 PM'
    }

    maindir =os.getcwd()
    Austin = DSS(dict_DSS, maindir)