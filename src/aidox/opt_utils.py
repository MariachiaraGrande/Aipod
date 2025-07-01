import numpy as np
from typing import Tuple
from sympy import sympify

from ._utils import setup_logger

LOG = setup_logger('OPT', 'DEBUG')


def callback_tol(
        tol_loss: float, delta_tol: float,
        check_step_delta: int, candidate, opt, loss
):
    """
    Callback for tol evaluation

    :param tol_loss: tolerance for loss value
    :param delta_tol: tolerance for delta loss decrease
    :param check_step_delta: number of steps to be checked for delta decrease
    :param candidate: optimizer candidate
    :param opt: optimizer instance
    :param loss: loss value

    """
    opt._loss_hist = getattr(opt, '_loss_hist', [1e10])
    opt._loss_hist.append(loss)

    # check if decrement in loss is lower than passed
    checks = min(check_step_delta, len(opt._loss_hist))

    if np.abs(loss) < tol_loss:
        LOG.debug('Reached tol')
        opt._done = True

    else:
        if (np.abs((np.diff(opt._loss_hist[-checks:])) / (np.abs(opt._loss_hist)[-checks + 1:])) < delta_tol).all():
            LOG.debug(f'Not decresing for {checks} steps')
            opt._done = True



def const_type(c: str) -> str:
    """
    Build constraint type base on formulation

    :param c: constraint
    :return:  type of constraint

    """
    if '>' in c:
        return 'up'
    elif '<' in c:
        return 'low'
    elif '=' in c and "<" not in c and ">" not in c:
        return 'eq'
    else:
        raise ValueError('Cannot parse constraint formula')


def process_formula(formula: str, c_type: str, idx: int) -> str:
    """
    Process constraint formula and flip them if needed in order to get a standard form
    of equality constraint (or upper than inequality that will be used with slack variables)

    :param formula: input formula
    :param c_type: type of constraint
    :param idx: index slack
    :return: updated formula

    """
    if c_type == 'eq':
        f_parts = formula.split(('='))
        assert len(f_parts) == 2, f'Wrong splitting formula for constraint {formula}'
        formula_upd = sympify(f_parts[0]) + sympify(f_parts[-1]) * -1
    else:
        if '<' in formula:
            # flip all and change sign
            f_parts = formula.replace('=', '').split('<')
            assert len(f_parts) == 2, f'Wrong splitting formula for constraint {formula}'
            formula_upd = sympify(f_parts[0]) * -1 + sympify(f_parts[1]) - sympify(f'S_{idx}**2')
        elif '>' in formula:
            # add slack and change sign
            f_parts = formula.replace('=', '').split('>')
            assert len(f_parts) == 2, f'Wrong splitting formula for constraint {formula}'
            formula_upd = sympify(f_parts[0]) + sympify(f_parts[1]) * -1 - sympify(f'S_{idx}**2')
        else:
            raise ValueError('Cannot invert inequality constraint')
    return formula_upd


def build_constraints(params: dict) -> Tuple[dict, dict]:
    """
    Build constraint from parameters

    :param params: input constraint parameter
    :return const_dict: update constraint dict
    :return slacks: slack variables dict

    """
    const_dict = {}
    slacks = {}
    for idx, c in enumerate(params):
        c_type = const_type(c['formula'])
        formula = process_formula(c['formula'], c_type, idx)
        const_dict[idx] = {
            'formula': formula,
            'tol': c['tol'],
            'c_type': c_type,
            'coeff': formula.as_coeff_add()
        }
        if c_type != 'eq':
            slacks[idx] = f'S_{idx}'
    return const_dict, slacks



class LagrangianHandler:
    """ Handler for Lagrangian Function (optimization loss + constraints)"""

    def __init__(self, mu: float, lambda_init: float, constraints: dict, opt_vars: dict, out_vars:dict=None):
        self.mu = mu
        self.eta = 1 / (mu ** 0.1)
        self.eta_k = 1 / (mu ** 0.1)
        self.omega = 1 / mu
        self.omega_k = 1 / mu
        self.constraints = constraints
        self.lambda_ = {idx: lambda_init for idx, _ in enumerate(self.constraints)}
        self.map = {x: y['id'] for x, y in opt_vars.items()}
        if out_vars is not None:
            self.map.update({x: y['id'] for x, y in out_vars.items()})
        self.lag_loss = None

    def build_lag_loss(self):
        """
        Build lagrangian loss from constraint parameters

        """
        lag_loss = []
        for idx, c in self.constraints.items():
            c_formula = - (self.lambda_[idx] * sympify(c['formula'])) + (self.mu * sympify(c['formula']) ** 2) / 2
            lag_loss.append(str(c_formula))
        if lag_loss:
            add_loss = '+'.join(lag_loss)
        else:
            add_loss = 0
        self.lag_loss = add_loss

    def eval_cost_const(self, param_value: dict) -> dict:
        """
        Eval constraint cost using passed mapping value of parameters

        :param param_value: value of parameters to be used in constraint formula
        :return cost_c: cost of constraint violation
        """
        cost_c = {}
        for c, c_params in self.constraints.items():
            cost_c[c] = float(sympify(c_params['formula']).evalf(subs=param_value))
        return cost_c

    def update_params(self, cost_c: dict, increse_penalty: bool):
        """
        Update of optimization parameters based on LANCELOT algorithm

        :param cost_c: cost of constraint violation
        :param increse_penalty: flag than enable penalty increase according to LANCELOT

        """
        if increse_penalty:
            self.mu = self.mu * 100
            self.eta_k = 1 / (self.mu ** 0.1)
            self.omega_k = 1 / self.mu
        else:
            self.lambda_ = {idx: lambda_i - self.mu * cost_c[idx] for idx, lambda_i in enumerate(self.lambda_)}
            self.eta_k = self.eta_k / (self.mu ** 0.9)
            self.omega_k = self.omega_k / self.mu
