from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
from numpy.linalg import inv
import time
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

class DA_OBD(object):

    """ Notations:
    Bnode - nodes that has located with battery
    Snode - slack nodes
    NonSnode - nonslack nodes
    """

    def __init__(self, dict_DA_OBD):
        self.V0 = dict_DA_OBD['V0']
        self.Bnode = dict_DA_OBD['list_of_Bnode']
        self.Iteration = 0
        self.model = None

    def create_model(self, Param_dict):

        model = ConcreteModel("Retail pricing")

        ##################################### Sets
        model.Time = RangeSet(0, 23)
        model.TimeX = RangeSet(0, 22)
        model.XTime = RangeSet(1, 23)
        model.NonSnode = RangeSet(0, 88)
        model.Snode = RangeSet(0, 2)
        model.Bnode = self.Bnode

        ##################################### Low-level problem parameters
        model.C = Param(model.NonSnode, initialize=Param_dict['Capacity'])
        model.SOCmax = Param(model.NonSnode, initialize=Param_dict['MaxSOC'])
        model.SOCmin = Param(model.NonSnode, initialize=Param_dict['MinSOC'])
        model.SOCinit = Param(model.NonSnode, initialize=Param_dict['InitSOC'])
        model.SOCend = Param(model.NonSnode, initialize=Param_dict['EndSOC'])
        model.Rmax = Param(model.NonSnode, initialize=Param_dict['MaxChargeRate'])
        model.Rmin = Param(model.NonSnode, initialize=Param_dict['MinChargeRate'])
        model.lc = Param(model.NonSnode, initialize=Param_dict['ChargeLoss'])
        model.ld = Param(model.NonSnode, initialize=Param_dict['DischargeLoss'])
        model.Dlifetime = Param(model.NonSnode, initialize=Param_dict['LifetimeDiscount'])
        model.M = Param(initialize=1000)
        model.subsidy = Param(initialize=0.5)

        def PL_init(model, t, i):
            print(-Param_dict['ActiveLoad'].T[t, i])
            return -Param_dict['ActiveLoad'].T[t, i]

        def QL_init(model, t, i):
            return -Param_dict['ReactiveLoad'].T[t, i]

        ##################################### High-level problem parameters
        model.P_L = Param(model.Time, model.NonSnode, initialize=PL_init)
        model.Q_L = Param(model.Time, model.NonSnode, initialize=QL_init)
        model.WP_p = Param(model.Time, initialize=Param_dict['WholesaleP'])
        model.RP_E0 = Param(model.Time, initialize=Param_dict['RetailE0'])

        def Breal_init(model, t, j, i):
            return self.Breal_24[t, j, i]

        def Bimag_init(model, t, j, i):
            return self.Bimag_24[t, j, i]

        def C1_init(model, t, j, i):
            return self.C1_24[t, j, i]

        def C2_init(model, t, j, i):
            return self.C2_24[t, j, i]

        def breal_init(model, t, j):
            return self.breal_24[t, j][0]

        def bimag_init(model, t, j):
            return self.bimag_24[t, j][0]

        def c_init(model, t, i):
            return self.c_24[t, i][0]

        model.Breal = Param(model.Time, model.Snode, model.NonSnode, initialize=Breal_init, mutable=True)
        model.Bimag = Param(model.Time, model.Snode, model.NonSnode, initialize=Bimag_init, mutable=True)
        model.C1 = Param(model.Time, model.NonSnode, model.NonSnode, initialize=C1_init, mutable=True)
        model.C2 = Param(model.Time, model.NonSnode, model.NonSnode, initialize=C2_init, mutable=True)
        model.breal = Param(model.Time, model.Snode, initialize=breal_init, mutable=True)
        model.bimag = Param(model.Time, model.Snode, initialize=bimag_init, mutable=True)
        model.c = Param(model.Time, model.NonSnode, initialize=c_init, mutable=True)

        ##################################### Low-level problem primal variables
        model.Pin = Var(model.Time, model.Bnode, domain=Reals)
        model.Pout = Var(model.Time, model.Bnode, domain=Reals)
        model.SOC = Var(model.Time, model.Bnode, domain=Reals)
        model.R = Var(model.Time, model.Bnode, domain=Reals)

        ##################################### Low-level problem dual variables
        model.lambda_SOC = Var(model.Time, model.Bnode, domain=Reals)
        model.lambda_R = Var(model.Time, model.Bnode, domain=Reals)
        model.mu_SOC_max = Var(model.Time, model.Bnode, domain=Reals)
        model.mu_SOC_min = Var(model.Time, model.Bnode, domain=Reals)
        model.mu_R_max = Var(model.Time, model.Bnode, domain=Reals)
        model.mu_R_min = Var(model.Time, model.Bnode, domain=Reals)
        model.mu_Pin_min = Var(model.Time, model.Bnode, domain=Reals)
        model.mu_Pout_min = Var(model.Time, model.Bnode, domain=Reals)

        ##################################### Big-M method binary variables
        model.w_SOC_max = Var(model.Time, model.Bnode, within=Binary)
        model.w_SOC_min = Var(model.Time, model.Bnode, within=Binary)
        model.w_R_max = Var(model.Time, model.Bnode, within=Binary)
        model.w_R_min = Var(model.Time, model.Bnode, within=Binary)
        model.w_Pin_min = Var(model.Time, model.Bnode, within=Binary)
        model.w_Pout_min = Var(model.Time, model.Bnode, within=Binary)

        ##################################### High-level problem primal variables
        model.RP_p = Var(model.Time, domain=Reals)
        model.Vmag = Var(model.Time, model.NonSnode, domain=Reals)
        model.Vreal = Var(model.Time, model.NonSnode, domain=Reals)
        model.Vimag = Var(model.Time, model.NonSnode, domain=Reals)
        model.P0 = Var(model.Time, model.Snode, domain=Reals)
        model.Q0 = Var(model.Time, model.Snode, domain=Reals)

        model.pprint()

        # ------------------------------------ Stationarity -------------------------------------
        def stationarity_Pin(model, t, i):
            return model.RP_p[t] + model.lambda_SOC[t, i] * (1.0 - model.lc[i]) * 100 / model.C[i] \
                   + model.lambda_R[t, i] * (1.0 - model.lc[i]) - model.mu_Pin_min[t, i] == 0.

        def stationarity_Pout(model, t, i):
            return - model.RP_p[t] - model.lambda_SOC[t, i] * 100 / ((1.0-model.ld[i])*model.C[i]) \
                   + model.lambda_R[t, i] / (1.0 - model.ld[i]) - model.mu_Pout_min[t, i] == 0.

        def stationarity_SOC(model, t, i):
            return model.lambda_SOC[t+1, i] - model.lambda_SOC[t, i] + model.mu_SOC_max[t, i] - model.mu_SOC_min[t, i] \
                   == 0.

        def stationarity_SOC_24(model, i):
            return -model.lambda_SOC[23, i] + model.mu_SOC_max[23, i] - model.mu_SOC_min[23, i] == 0.

        def stationarity_R(model, t, i):
            return model.Dlifetime[i] - model.lambda_R[t, i] + model.mu_R_max[t, i] - model.mu_R_min[t, i] == 0.

        model.SPin = Constraint(model.Time, model.Bnode, rule=stationarity_Pin)
        model.SPout = Constraint(model.Time, model.Bnode, rule=stationarity_Pout)
        model.SSOC = Constraint(model.TimeX, model.Bnode, rule=stationarity_SOC)
        model.SSOC_24 = Constraint(model.Bnode, rule=stationarity_SOC_24)
        model.SR = Constraint(model.Time, model.Bnode, rule=stationarity_R)

        # ------------------------------------ Complementary slackness -------------------------------------
        def cs_SOC_min_1_lower(model, t, i):
            return model.mu_SOC_min[t, i] >= 0

        def cs_SOC_min_1_upper(model, t, i):
            return model.w_SOC_min[t, i] * model.M - model.mu_SOC_min[t, i] >= 0

        def cs_SOC_min_2_lower(model, t, i):
            return model.SOC[t, i] - model.SOCmin[i] >= 0

        def cs_SOC_min_2_upper(model, t, i):
            return (1-model.w_SOC_min[t, i]) * model.M - model.SOC[t, i] + model.SOCmin[i] >= 0

        def cs_SOC_min_24_1_lower(model, i):
            return model.mu_SOC_min[23, i] >= 0

        def cs_SOC_min_24_1_upper(model, i):
            return model.w_SOC_min[23, i] * model.M - model.mu_SOC_min[23, i] >= 0

        def cs_SOC_min_24_2_lower(model, i):
            return model.SOC[23, i] - model.SOCend[i] >= 0

        def cs_SOC_min_24_2_upper(model, i):
            return (1 - model.w_SOC_min[23, i]) * model.M - model.SOC[23, i] + model.SOCend[i] >= 0

        def cs_SOC_max_1_lower(model, t, i):
            return model.mu_SOC_max[t, i] >= 0

        def cs_SOC_max_1_upper(model, t, i):
            return model.w_SOC_max[t, i] * model.M - model.mu_SOC_max[t, i] >= 0

        def cs_SOC_max_2_lower(model, t, i):
            return model.SOCmax[i] - model.SOC[t, i] >= 0

        def cs_SOC_max_2_upper(model, t, i):
            return (1 - model.w_SOC_max[t, i]) * model.M - model.SOCmax[i] + model.SOC[t, i] >= 0

        def cs_R_min_1_lower(model, t, i):
            return model.mu_R_min[t, i] >= 0

        def cs_R_min_1_upper(model, t, i):
            return model.w_R_min[t, i] * model.M - model.mu_R_min[t, i] >= 0

        def cs_R_min_2_lower(model, t, i):
            return model.R[t, i] - model.Rmin[i] >= 0

        def cs_R_min_2_upper(model, t, i):
            return (1 - model.w_R_min[t, i]) * model.M - model.R[t, i] + model.Rmin[i] >= 0

        def cs_R_max_1_lower(model, t, i):
            return model.mu_R_max[t, i] >= 0

        def cs_R_max_1_upper(model, t, i):
            return model.w_R_max[t, i] * model.M - model.mu_R_max[t, i] >= 0

        def cs_R_max_2_lower(model, t, i):
            return model.Rmax[i] - model.R[t, i] >= 0

        def cs_R_max_2_upper(model, t, i):
            return (1 - model.w_R_max[t, i]) * model.M - model.Rmax[i] + model.R[t, i] >= 0

        def cs_Pin_min_1_lower(model, t, i):
            return model.mu_Pin_min[t, i] >= 0

        def cs_Pin_min_1_upper(model, t, i):
            return model.w_Pin_min[t, i] * model.M - model.mu_Pin_min[t, i] >= 0

        def cs_Pin_min_2_lower(model, t, i):
            return model.Pin[t, i] >= 0

        def cs_Pin_min_2_upper(model, t, i):
            return (1 - model.w_Pin_min[t, i]) * model.M - model.Pin[t, i] >= 0

        def cs_Pout_min_1_lower(model, t, i):
            return model.mu_Pout_min[t, i] >= 0

        def cs_Pout_min_1_upper(model, t, i):
            return model.w_Pout_min[t, i] * model.M - model.mu_Pout_min[t, i] >= 0

        def cs_Pout_min_2_lower(model, t, i):
            return model.Pout[t, i] >= 0

        def cs_Pout_min_2_upper(model, t, i):
            return (1 - model.w_Pout_min[t, i]) * model.M - model.Pout[t, i] >= 0

        model.CSSOClower_1_lower = Constraint(model.TimeX, model.Bnode, rule=cs_SOC_min_1_lower)
        model.CSSOClower_2_lower = Constraint(model.TimeX, model.Bnode, rule=cs_SOC_min_2_lower)
        model.CSSOClower_24_1_lower = Constraint(model.Bnode, rule=cs_SOC_min_24_1_lower)
        model.CSSOClower_24_2_lower = Constraint(model.Bnode, rule=cs_SOC_min_24_2_lower)
        model.CSSOCupper_1_lower = Constraint(model.Time, model.Bnode, rule=cs_SOC_max_1_lower)
        model.CSSOCupper_2_lower = Constraint(model.Time, model.Bnode, rule=cs_SOC_max_2_lower)
        model.CSRlower_1_lower = Constraint(model.Time, model.Bnode, rule=cs_R_min_1_lower)
        model.CSRlower_2_lower = Constraint(model.Time, model.Bnode, rule=cs_R_min_2_lower)
        model.CSRupper_1_lower = Constraint(model.Time, model.Bnode, rule=cs_R_max_1_lower)
        model.CSRupper_2_lower = Constraint(model.Time, model.Bnode, rule=cs_R_max_2_lower)
        model.CSPinlower_1_lower = Constraint(model.Time, model.Bnode, rule=cs_Pin_min_1_lower)
        model.CSPinlower_2_lower = Constraint(model.Time, model.Bnode, rule=cs_Pin_min_2_lower)
        model.CSPoutlower_1_lower = Constraint(model.Time, model.Bnode, rule=cs_Pout_min_1_lower)
        model.CSPoutlower_2_lower = Constraint(model.Time, model.Bnode, rule=cs_Pout_min_2_lower)

        model.CSSOClower_1_upper = Constraint(model.TimeX, model.Bnode, rule=cs_SOC_min_1_upper)
        model.CSSOClower_2_upper = Constraint(model.TimeX, model.Bnode, rule=cs_SOC_min_2_upper)
        model.CSSOClower_24_1_upper = Constraint(model.Bnode, rule=cs_SOC_min_24_1_upper)
        model.CSSOClower_24_2_upper = Constraint(model.Bnode, rule=cs_SOC_min_24_2_upper)
        model.CSSOCupper_1_upper = Constraint(model.Time, model.Bnode, rule=cs_SOC_max_1_upper)
        model.CSSOCupper_2_upper = Constraint(model.Time, model.Bnode, rule=cs_SOC_max_2_upper)
        model.CSRlower_1_upper = Constraint(model.Time, model.Bnode, rule=cs_R_min_1_upper)
        model.CSRlower_2_upper = Constraint(model.Time, model.Bnode, rule=cs_R_min_2_upper)
        model.CSRupper_1_upper = Constraint(model.Time, model.Bnode, rule=cs_R_max_1_upper)
        model.CSRupper_2_upper = Constraint(model.Time, model.Bnode, rule=cs_R_max_2_upper)
        model.CSPinlower_1_upper = Constraint(model.Time, model.Bnode, rule=cs_Pin_min_1_upper)
        model.CSPinlower_2_upper = Constraint(model.Time, model.Bnode, rule=cs_Pin_min_2_upper)
        model.CSPoutlower_1_upper = Constraint(model.Time, model.Bnode, rule=cs_Pout_min_1_upper)
        model.CSPoutlower_2_upper = Constraint(model.Time, model.Bnode, rule=cs_Pout_min_2_upper)

        # ------------------------------------ Primal feasibility -------------------------------------
        def SOC_update_rule_1(model, i):
            return model.SOC[0, i] == model.SOCinit[i] + model.Pin[0, i] * (1.0 - model.lc[i]) * 100 / model.C[i] - \
                   model.Pout[0, i] * 100 / ((1.0 - model.ld[i]) * model.C[i])

        def SOC_update_rule(model, t, i):
            return model.SOC[t, i] == model.SOC[t - 1, i] + model.Pin[t, i] * (1.0 - model.lc[i]) * 100 / model.C[i] - \
                   model.Pout[t, i] * 100 / ((1.0 - model.ld[i]) * model.C[i])

        def SOC_inequality(model, t, i):
            return (model.SOCmin[i], model.SOC[t, i], model.SOCmax[i])

        def SOC_inequality_24(model, i):
            return (model.SOCend[i], model.SOC[23, i], model.SOCmax[i])

        def Rate_equality(model, t, i):
            return model.R[t, i] == model.Pin[t, i]*(1.0-model.lc[i]) + model.Pout[t, i]/(1.0-model.ld[i])

        def Rate_inequality(model, t, i):
            return (model.Rmin[i], model.R[t, i], model.Rmax[i])

        def Pin_inequality(model, t, i):
            return model.Pin[t, i] >= 0

        def Pout_inequality(model, t, i):
            return model.Pout[t, i] >= 0

        model.SOCEqu_1 = Constraint(model.Bnode, rule=SOC_update_rule_1)
        model.SOCEqu = Constraint(model.XTime, model.Bnode, rule=SOC_update_rule)
        model.SOCInequ = Constraint(model.TimeX, model.Bnode, rule=SOC_inequality)
        model.SOCInequ_24 = Constraint(model.Bnode, rule=SOC_inequality_24)
        model.RateEqu = Constraint(model.Time, model.Bnode, rule=Rate_equality)
        model.RateInequ = Constraint(model.Time, model.Bnode, rule=Rate_inequality)
        model.PinInequ = Constraint(model.Time, model.Bnode, rule=Pin_inequality)
        model.PoutInequ = Constraint(model.Time, model.Bnode, rule=Pout_inequality)

        # ------------------------------------ Dual feasibility -------------------------------------
        def dualfeas_muRupper(model, t, i):
            return model.mu_R_max[t, i] >= 0

        def dualfeas_muRlower(model, t, i):
            return model.mu_R_min[t, i] >= 0

        def dualfeas_muSOCupper(model, t, i):
            return model.mu_SOC_max[t, i] >= 0

        def dualfeas_muSOClower(model, t, i):
            return model.mu_SOC_min[t, i] >= 0

        def dualfeas_muPinlower(model, t, i):
            return model.mu_Pin_min[t, i] >= 0

        def dualfeas_muPoutlower(model, t, i):
            return model.mu_Pout_min[t, i] >= 0

        model.DFRlower = Constraint(model.Time, model.Bnode, rule=dualfeas_muRlower)
        model.DFRupper = Constraint(model.Time, model.Bnode, rule=dualfeas_muRupper)
        model.DFSOClower = Constraint(model.Time, model.Bnode, rule=dualfeas_muSOClower)
        model.DFSOCupper = Constraint(model.Time, model.Bnode, rule=dualfeas_muSOCupper)
        model.DFPinlower = Constraint(model.Time, model.Bnode, rule=dualfeas_muPinlower)
        model.DFPoutlower = Constraint(model.Time, model.Bnode, rule=dualfeas_muPoutlower)

        # ------------------------------------ Higher-level constraints -------------------------------------
        def P0_equality(model, t, j):
            return model.P0[t, j] == sum(model.Breal[t, j, k] * (model.Pout[t, k] - model.Pin[t, k])
                                         for k in model.Bnode) + model.breal[t, j]

        def Q0_equality(model, t, j):
            return model.Q0[t, j] == sum(model.Bimag[t, j, k] * (model.Pout[t, k] - model.Pin[t, k])
                                         for k in model.Bnode) + model.bimag[t, j]

        def Vmag_equality(model, t, i):
            return model.Vmag[t, i] == sum(model.C1[t, i, k] * (model.Pout[t, k] - model.Pin[t, k])
                                           for k in model.Bnode) + model.c[t, i]

        def Vmag_inequality(model, t, i):
            return (0.945, model.Vmag[t, i], 1.05)

        # model.P0Equ = Constraint(model.Time, model.Snode, rule=P0_equality)
        # model.Q0Equ = Constraint(model.Time, model.Snode, rule=Q0_equality)
        # model.VmagEqu = Constraint(model.Time, model.NonSnode, rule=Vmag_equality)
        # model.VmagInequ = Constraint(model.Time, model.NonSnode, rule=Vmag_inequality)

        # when not considering the network
        def power_balance(model, t):
            return sum(model.P_L[t, i] for i in model.NonSnode) + sum(model.Pin[t, i] for i in model.Bnode) == \
                   sum(model.P0[t, i] for i in model.Snode) + sum(model.Pout[t, i] for i in model.Bnode)

        def price_equality_P(model):
            return sum(model.RP_p[t] for t in model.Time) == 1.0

        def price_inequality_P(model, t):
            return 0.06 >= model.RP_p[t] >= 0

        model.powerbalance = Constraint(model.Time, rule=power_balance)
        model.priceEquP = Constraint(rule=price_equality_P)
        model.priceInequP = Constraint(model.Time, rule=price_inequality_P)

        # ------------------------------------ Objective function -------------------------------------
        def obj_expression(model):
            return sum(sum(model.RP_p[t]*model.P_L[t, k] for k in model.NonSnode) for t in model.Time)\
                   - sum(sum(model.WP_p[t] * model.P0[t, i] for i in model.Snode) for t in model.Time) - \
                   sum( - model.lambda_SOC[0, i] * model.SOCinit[i] + \
                       sum(model.mu_SOC_max[t, i] * model.SOCmax[i] for t in model.Time) - \
                       model.mu_SOC_min[23, i] * model.SOCend[i] - \
                       sum(model.mu_SOC_min[t, i] * model.SOCmin[i] for t in model.TimeX) + \
                       sum(model.mu_R_max[t, i] * model.Rmax[i] for t in model.Time) - \
                       sum(model.mu_R_min[t, i] * model.Rmin[i] for t in model.Time) + \
                       sum(model.Dlifetime[i] * model.R[t, i] for t in model.Time)
                       for i in model.Bnode)

        model.OBJ = Objective(rule=obj_expression, sense=maximize)

        # Obtain dual solutions from first solve and send to warm start
        model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        print("Model created successifully")
        self.model = model

    def solve(self):

        opt = SolverFactory('gurobi')
        opt.set_options("IntFeasTol=10e-9")

        print('solve')
        result = opt.solve(self.model, tee=True)
        print(result)

    def get_Ymatrix(self):
        # Reading Y matrix
        Yreal = pd.read_csv('Data/Yreal.csv', index_col=0).values
        Yimag = pd.read_csv('Data/Yimag.csv', index_col=0).values
        Y = np.array(Yreal + np.multiply(1j, Yimag))
        self.Y00 = Y[:3, :3]
        self.Y0L = Y[:3, 3:]
        self.YL0 = Y[3:, :3]
        self.YLL = Y[3:, 3:]
        self.YLLinv = inv(self.YLL)

    def get_coefficients(self, Vhat):
        A_24 = []
        Breal_24 = []
        Bimag_24 = []
        C1_24 = []
        C2_24 = []
        a_24 = []
        breal_24 = []
        bimag_24 = []
        c_24 = []

        for t in range(24):
            Vhat_ = Vhat[:, t:t + 1]

            # Calculating coefficient matrixes
            w = -(np.matmul(np.matmul(self.YLLinv, self.YL0), self.V0))
            W = np.diag(w.T[0])

            A = np.matmul(self.YLLinv, inv(np.diag(np.conjugate(Vhat_.T[0]))))
            a = w + np.matmul(A, self.PL[t:t+1, :].T) + \
                np.matmul(np.multiply(A, np.array([0 - 1j])), self.QL[t:t+1, :].T)

            B = np.matmul(np.matmul(np.diag(self.V0.T[0]), np.conjugate(self.Y0L)), np.conjugate(A))
            b = np.matmul(np.diag(self.V0.T[0]), (np.matmul(np.conjugate(self.Y00), np.conjugate(self.V0))
                                                  + np.matmul(np.conjugate(self.Y0L), np.conjugate(w)))) \
                + np.matmul(B, self.PL[t:t+1, :].T) + np.matmul(np.multiply(B, np.array([1j])), self.QL[t:t+1, :].T)

            C1 = np.matmul(np.absolute(W), np.matmul(inv(W), A).real)
            C2 = np.matmul(np.absolute(W), np.matmul(inv(W), np.multiply(-1j, A)).real)
            c = np.absolute(w) + np.matmul(C1, self.PL[t:t+1, :].T) + np.matmul(C2, self.QL[t:t+1, :].T)

            A_24.append(A)
            a_24.append(a)
            Breal_24.append(B.real)
            Bimag_24.append(B.imag)
            breal_24.append(b.real)
            bimag_24.append(b.imag)
            C1_24.append(C1)
            C2_24.append(C2)
            c_24.append(c)

        np.savetxt('Results/C1.csv', C1, delimiter=',')
        np.savetxt('Results/C2.csv', C2, delimiter=',')

        self.A_24 = np.array(A_24)
        self.a_24 = np.array(a_24)
        self.Breal_24 = np.array(Breal_24)
        self.Bimag_24 = np.array(Bimag_24)
        self.C1_24 = np.array(C1_24)
        self.C2_24 = np.array(C2_24)
        self.breal_24 = np.array(breal_24)
        self.bimag_24 = np.array(bimag_24)
        self.c_24 = np.array(c_24)


    def read_configure(self):

        Param_list = ['Capacity', 'MaxSOC', 'MinSOC', 'InitSOC', 'EndSOC', 'MaxChargeRate', 'MinChargeRate',
                      'ChargeLoss', 'DischargeLoss', 'LifetimeDiscount', 'ActiveLoad', 'ReactiveLoad', 'WholesaleP',
                      'WholesaleQ', 'RetailE0']
        Param_dict = dict()

        for Param in Param_list:
            if Param == 'ActiveLoad':
                Param_dict[Param] = pd.read_csv('Data/P.csv', index_col=0).values
            elif Param == 'ReactiveLoad':
                Param_dict[Param] = pd.read_csv('Data/Q.csv', index_col=0).values
            else:
                Param_dict[Param] = \
                pd.DataFrame(pd.read_csv('Configuration/' + Param + '.csv', index_col=0)[Param].tolist(),
                             index=range(len(pd.read_csv('Configuration/' + Param + '.csv', index_col=0)[
                                                 Param].tolist()))).to_dict()[0]
                # Param_dict[Param] = \
                # pd.DataFrame(pd.read_csv('Configuration_nonidentical/' + Param + '.csv', index_col=0)[Param].tolist(),
                #              index=range(len(pd.read_csv('Configuration_nonidentical/' + Param + '.csv', index_col=0)[
                #                                  Param].tolist()))).to_dict()[0]
        return Param_dict

    def successive_solve(self, num):
        Param_dict = self.read_configure()
        self.PL = Param_dict['ActiveLoad'].T
        self.QL = Param_dict['ReactiveLoad'].T
        self.get_Ymatrix()
        for i in range(num):
            if i == 0:
                w = -(np.matmul(np.matmul(self.YLLinv, self.YL0), self.V0))
                Vhat = np.repeat(w, 24, axis=1)
            else:
                Pin = np.zeros((24, 89))
                Pout = np.zeros((24, 89))
                for t in self.model.Time:
                    for i in self.model.Bnode:
                        Pin[t, i] = value(self.model.Pin[t, i])
                        Pout[t, i] = value(self.model.Pout[t, i])
                    Vhat[:, t:t+1] = np.matmul(self.A_24[t:t+1, :, :], Pout[t:t+1, :].T - Pin[t:t+1, :].T) + self.a_24[t:t+1, :]
            self.get_coefficients(Vhat)
            self.create_model(Param_dict)
            self.solve()
            # self.store_Vhat() #TODO
        self.store_results()

    def non_successive_solve(self, num):
        Param_dict = self.read_configure()
        self.PL = Param_dict['ActiveLoad'].T
        self.QL = Param_dict['ReactiveLoad'].T
        self.create_model(Param_dict)
        self.solve()
        self.store_results()

    def store_results(self):
        Pin = np.zeros((24, 89))
        Pout = np.zeros((24, 89))
        SOC = np.zeros((24, 89))
        R = np.zeros((24, 89))
        B_revenue = np.zeros((24, 89))
        RP_B = []
        RP_E = []
        WP_p = []
        RP_E0 = []
        P0 = np.zeros((24, 3))
        Q0 = np.zeros((24, 3))
        PL = np.zeros((24, 89))
        Vmag = np.zeros((24, 89))

        for t in self.model.Time:
            for i in self.model.Bnode:
                Pin[t, i] = value(self.model.Pin[t, i])
                Pout[t, i] = value(self.model.Pout[t, i])
                SOC[t, i] = value(self.model.SOC[t, i])
                R[t, i] = value(self.model.R[t, i])
                B_revenue[t, i] = (value(self.model.Pout[t, i]) - value(self.model.Pin[t, i])) * value(self.model.RP_p[t])

        for t in self.model.Time:
            RP_B.append(value(self.model.RP_p[t]))
            WP_p.append(value(self.model.WP_p[t]))
            RP_E0.append(value(self.model.RP_E0[t]))

        for t in self.model.Time:
            for i in self.model.NonSnode:
                PL[t, i] = value(self.model.P_L[t, i])
                # Vmag[t, i] = value(self.model.Vmag[t, i])
            for i in self.model.Snode:
                P0[t, i] = value(self.model.P0[t, i])
                # Q0[t, i] = value(self.model.Q0[t, i])

        # for t in self.model.Time:
        #     RP_E.append(value(self.model.subsidy) * RP_E0[t] + (1 - value(self.model.subsidy)) * (
        #                 RP_B[t] * sum([Pout[t, i] - Pin[t, i] for i in self.model.Bnode]) + \
        #                 WP_p[t] * sum([P0[t, j] for j in self.model.Snode])) \
        #                 / sum(PL[t, k] for k in self.model.NonSnode))

        np.savetxt('Results/Pin.csv', Pin, delimiter=',')
        np.savetxt('Results/Pout.csv', Pout, delimiter=',')
        np.savetxt('Results/RP_B.csv', RP_B, delimiter=',')
        # np.savetxt('Results/RP_E.csv', RP_E, delimiter=',')
        np.savetxt('Results/WP_p.csv', WP_p, delimiter=',')
        np.savetxt('Results/P0.csv', P0, delimiter=',')
        np.savetxt('Results/Q0.csv', Q0, delimiter=',')
        np.savetxt('Results/Vmag.csv', Vmag, delimiter=',')
        np.savetxt('Results/SOC.csv', SOC, delimiter=',')
        np.savetxt('Results/R.csv', R, delimiter=',')

        # Plotting results
        datas = pd.DataFrame({'Wholesale Price': WP_p, 'Retail Price': RP_B, 'System Load': [ x/20000 for x in np.sum(PL, axis=1).tolist()],
                              'Hour': self.model.Time}, index=self.model.Time)

        for i in range(len(self.Bnode)):
            Pin_B = np.zeros((len(self.model.Time), 1))
            Pout_B = np.zeros((len(self.model.Time), 1))
            for j in range(i, len(self.Bnode)):
                Pin_B = Pin_B + Pin[:, self.Bnode[j]:self.Bnode[j]+1]
                print(np.shape(Pin[:, self.Bnode[j]:self.Bnode[j]+1]))
                Pout_B = Pout_B + Pout[:, self.Bnode[j]:self.Bnode[j]+1]
            datas['Battery %d' %i] = Pin_B - Pout_B

        for i in self.Bnode:
            print(i)
            print("Revenue:", sum(B_revenue[:, i: i+1].T[0]))

        print("Objective function", value(self.model.OBJ))
        print("Overall cost", sum(WP_p[t] * sum( value(self.model.P0[t, i]) for i in self.model.Snode) for t in self.model.Time)
              + sum(RP_B[t] * sum( value(self.model.Pout[t, i]) - value(self.model.Pin[t, i]) for i in self.model.Bnode) for t in self.model.Time))
        print("Overall payment", sum(RP_B[t] * sum( value(self.model.P_L[t, i]) for i in self.model.NonSnode)
                                  for t in self.model.Time))
        print("Total revenue", sum(RP_B[t] * sum( value(self.model.Pout[t, i]) - value(self.model.Pin[t, i]) for i in self.model.Bnode) for t in self.model.Time))
        print("Discounted revenue", sum(
            RP_B[t] * sum(value(self.model.Pout[t, i]) - value(self.model.Pin[t, i]) for i in self.model.Bnode) for t in
            self.model.Time)
              - sum(value(self.model.Dlifetime[i]) * sum(value(self.model.R[t, i]) for t in self.model.Time) for i in
                    self.model.Bnode))
        print("Original cost", sum(WP_p[t] * sum( value(self.model.P_L[t, i]) for i in self.model.NonSnode) for t in self.model.Time))
        sns.color_palette("Set1", n_colors=8, desat=.5)
        sns.set(rc={'figure.figsize': (7, 4.5)})
        sns.set_style("ticks", {"xtick.major.size": 10, "ytick.major.size": 10})
        sns.set_style("dark")
        my_colors = ['b', 'green', 'y', 'pink', 'orange', 'cyan', 'darkgrey', 'black', 'magenta', 'burlywood',
                     'saddlebrown', 'tomato', 'dimgrey', 'indianred', 'yellowgreen', 'mediumpurple', 'palegoldenrod',
                     'lavender', 'khaki', 'darkkhaki']

        for i in range(len(self.Bnode)):
            p = sns.barplot(x='Hour', y='Battery %d' %i, data=datas, color=my_colors[i], saturation=0.4)
        ax2 = p.twinx()
        q = sns.lineplot(x='Hour', y='Wholesale Price', ax=ax2, data=datas, label=r'Wholesale price', color='salmon', marker='^')
        q = sns.lineplot(x='Hour', y='Retail Price', ax=ax2, data=datas, label=r'Retail price', color='lightgreen', marker='^')
        # q = sns.lineplot(x='Hour', y='Retail Energy Price', ax=ax2, data=datas, label=r'Retail energy price',
        #                  color='lightseagreen', marker='^')
        q = sns.lineplot(x='Hour', y='System Load', ax=ax2, data=datas, label=r'System load', color='gray', marker='o')
        p.set_xlabel('Hour', fontsize=13)
        p.set_ylabel('Charging/Discharing Quantity (KWh)', fontsize=13)
        q.set_ylabel('Price ($/KWh), Load (x20MWh)', fontsize=13)
        p.set(ylim=(-1100, 1100))
        q.set(ylim=(-0.002, 0.062))

        q.legend(loc="lower right", fontsize=10)
        plt.show()


if __name__ == "__main__":
    start = time.time()
    dict_DA_OBD = {
        'V0': np.array([[1 + 0j], [-0.5 - 0.8666j], [-0.5 + 0.8666j]]),
        'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64, 66, 68, 69, 70, 73, 76, 77, 82],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64, 66, 68, 69, 70, 73, 76, 77],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64, 66, 68, 69, 70, 73, 76],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64, 66, 68, 69, 70, 73],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64, 66, 68, 69, 70],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64, 66, 68, 69],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64, 66, 68],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64, 66],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63, 64],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62, 63],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61, 62],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58, 61],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55, 58],
        # 'list_of_Bnode': [16, 17, 18, 45, 48, 52, 55],
        # 'list_of_Bnode': [16, 17, 55, 45, 48, 52],
        # 'list_of_Bnode': [16, 17, 18, 45, 48],
        # 'list_of_Bnode': [16, 17, 55, 45],
        # 'list_of_Bnode': [16, 17, 55],
        # 'list_of_Bnode': [16, 55],
        # 'list_of_Bnode': [55]
    }

    IEEE123 = DA_OBD(dict_DA_OBD)
    IEEE123.successive_solve(10)
    end = time.time()
    print('Time lapse is: %f', end-start)