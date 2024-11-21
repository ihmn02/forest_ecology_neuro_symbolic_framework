import torch
from rule_templates import Rule_p_imp_q, Rule_p_iff_q


class Rule1(Rule_p_iff_q):
    '''
    prediction is a tree if and only if its area is less than the site mean ITC
    '''
    def __init__(self, rule_name, rule_fol, lmbda):
         super().__init__(rule_name, rule_fol, lmbda)

    def get_val(self):
        out = self.eval(self.rule_fxn_output, self.tree)
        return out

    def rule_fxn(self, input_tensor, **kwargs):
        device = input_tensor[0].device
        sig = torch.nn.Sigmoid()
        bb_areas = kwargs['bb_areas']

        #return the index of bboxes with areas greater than X
        res = sig(0.5 * (400. - bb_areas.to(device)))
        return res


class Rule2(Rule_p_iff_q):
    '''
    a prediction is a tree if and only if its area is less than or equal to the area predicted from the H-CA allometry
    '''
    def __init__(self, rule_name, rule_fol, lmbda):
         super().__init__(rule_name, rule_fol, lmbda)

    def get_val(self):
        out = self.eval(self.rule_fxn_output, self.tree)
        return out

    def rule_fxn(self, input_tensor, **kwargs): 
        '''
        Adjust predictions based on bounding box chm max height
        '''
        device = input_tensor[0].device

        # y = b * x^(a)
        a = 0.87992
        b = 0.32658
        k_sig = 0.5

        bb_max_hts = kwargs['bb_max_hts']
        optim_area = b * (torch.pow(bb_max_hts, a))
        optim_area = torch.divide(optim_area, 0.01)

        sig = torch.nn.Sigmoid()

        # calculate the area of each bounding box
        bb_areas = kwargs['bb_areas']

        #return the index of bboxes with areas greater than X
        res = sig(k_sig * (optim_area - bb_areas))
        return res

