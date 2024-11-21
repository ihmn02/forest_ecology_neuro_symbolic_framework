import torch
from rule_templates import Rule_p_imp_not_q, Rule_p_imp_q


class Rule1(Rule_p_imp_not_q):
    '''
    trees taller than thr unlikely to be black oak
    thr: 46.0m
    '''
    def __init__(self, rule_name, rule_fol, lmbda, rule_ind=5):
         super().__init__(rule_name, rule_fol, lmbda)
         self.rule_ind = rule_ind

    def get_val(self):
        out = self.eval(self.rule_fxn_output, self.black_oak)
        return out

    def rule_fxn(self, input_tensor, **kwargs):
        device = input_tensor.device
        sig = torch.nn.Sigmoid()
        return sig(-1.e3*(-1 * torch.amax(input_tensor, dim=(1,2)) + self.thr)).reshape(-1, 1)

    def set_thr(self, thr):
        self.thr = thr


class Rule2(Rule_p_imp_not_q):
    '''
    trees taller than thr unlikley to be lodgepole pine
    thr: 53.2m
    '''
    def __init__(self, rule_name, rule_fol, lmbda, rule_ind=6):
         super().__init__(rule_name, rule_fol, lmbda)
         self.rule_ind = rule_ind

    def get_val(self):
        out = self.eval(self.rule_fxn_output, self.lodgepole_pine)
        return out

    def rule_fxn(self, input_tensor, **kwargs):
        device = input_tensor.device
        sig = torch.nn.Sigmoid()
        return sig(-1.e3*(-1 * torch.amax(input_tensor, dim=(1,2)) + self.thr)).reshape(-1, 1)

    def set_thr(self, thr):
        self.thr = thr

class Rule3(Rule_p_imp_not_q):
    '''
    trees at an elevation of less than thr are unlikely to be red fir
    thr: 207.2m
    '''
    def __init__(self, rule_name, rule_fol, lmbda, rule_ind=1):
         super().__init__(rule_name, rule_fol, lmbda)
         self.rule_ind = rule_ind

    def get_val(self):
        out = self.eval(self.rule_fxn_output, self.red_fir)
        return out

    def rule_fxn(self, input_tensor, **kwargs):
        device = input_tensor.device
        dem = kwargs['dem']
        dem = dem[:, 0, :, :] 
        sig = torch.nn.Sigmoid()
        return sig(1.e3*(-1 * torch.amax(dem, dim=(1,2)) + self.thr)).reshape(-1, 1)

    def set_thr(self, thr):
        self.thr = thr
