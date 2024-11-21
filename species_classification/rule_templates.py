import torch

class Rule_p_imp_q:
    """
    all(x1) p => q
    x1: [N, 1] tensor
    y1: [N, 1] tensor
    """
    def __init__(self, rule_name, rule_fol, lmbda):
        self.rule_name = rule_name
        self.rule_fol = rule_fol
        self.lmbda = lmbda

    def luk_or(self, arg1, arg2):
        # Lukaseiwicz OR
        # out = 'arg1 or arg2'
        arg1 = arg1.reshape(-1, 1)
        arg2 = arg2.reshape(-1, 1)
        out = torch.minimum(torch.ones(arg1.shape, dtype=torch.float).to(arg1.device), arg1 + arg2)
        out = out.reshape(-1, 1)
        return out

    def luk_and(self, arg1, arg2):
        # Lukaseiwicz AND
        # out = 'arg1 AND arg2'
        arg1 = arg1.reshape(-1, 1)
        arg2 = arg2.reshape(-1, 1)
        out = torch.maximum(torch.zeros(arg1.shape, dtype=torch.float).to(arg1.device), arg1 + arg2 - 1.0)
        out = out.reshape(-1, 1)
        return out

    def luk_imp(self, arg1, arg2):
        # Lukaseiwicz =>
        # out = 'arg1 => arg2'
        arg1 = arg1.reshape(-1, 1)
        arg2 = arg2.reshape(-1, 1)
        out = torch.minimum(torch.ones(arg1.shape, dtype=torch.float).to(arg1.device), 1.0 - arg1 + arg2)
        out = out.reshape(-1, 1)
        return out

    def luk_not(self, arg1):
        # Lukaseiwicz NOT
        # out = 'not(arg1)'
        arg1 = arg1.reshape(-1, 1)
        #out = torch.ones(arg1.shape, dtype=torch.float).to(arg1.device) - arg1
        out = 1.0 - arg1
        out = out.reshape(-1, 1)
        return out 

    def process_logits(self, arg1):
        # scales logit with sigmoid of softmax depending on shape
        out = None
        if arg1.shape[1] == 1:
            sig = torch.nn.Sigmoid()
            out = sig(arg1)
            out = out.reshape(-1, 1)
        elif arg1.shape[1] > 1:
            softmax = torch.nn.Softmax(dim=1)
            out = softmax(arg1)
        return out

    def generic_interface(self, preds, rule_fxn_output):
        """
        Purpose: Provides an interface between the model and the rule objects. Needs to be customized to the model.
        """
        # classes in order: ('white_fir', 'red_fir', 'incense_cedar', 'jeffrey_pine', 'sugar_pine', 'black_oak', 'lodgepole_pine', 'dead')
        preds_canned = self.process_logits(preds)

        self.white_fir = preds_canned[:, 0]
        self.red_fir = preds_canned[:, 1]
        self.incense_cedar = preds_canned[:, 2]
        self.jeffrey_pine = preds_canned[:, 3]
        self.sugar_pine = preds_canned[:, 4]
        self.black_oak = preds_canned[:, 5]
        self.lodgepole_pine = preds_canned[:, 6]
        self.dead = preds_canned[:, 7]

        self.rule_fxn_output = self.process_logits(rule_fxn_output)

    def eval(self, p, q):
        out = self.luk_imp(p, q)
        return out 

class Rule_p_imp_not_q(Rule_p_imp_q):
    """
    all(x) p ==> ~q
    """
    def eval(self, x, y):
        y_not = self.luk_not(y)
        out = self.luk_imp(x, y_not)
        return out


class Rule_p_imp_disj_q(Rule_p_imp_q):
    """
    all(x) p ==> q1 v q2 v q3 v ... v qN
    """
    def eval(self, p, q_list):
        assert len(q_list) > 1, print("Error disjunction list")
        q_disj = self.luk_or(q_list[0], q_list[1])         
        for q in q_list[2:]:
            q_disj = self.luk_or(q_disj, q)
        out = self.luk_imp(p, q_disj)
        return out       

class Rule_disj_p(Rule_p_imp_q):
    """
    all(x) p1 v p2 v p3 v ... v pN
    """
    def eval(self, p_list):
        assert len(p_list) > 1, print("Error disjunction list")
        p_disj = self.luk_or(p_list[0], p_list[1])
        for p in p_list[2:]:
            p_disj = self.luk_or(p_disj, p)
        out = p_disj
        return out

class Rule_not_p_imp_q(Rule_p_imp_q):
    """
    all(x) ~p ==> q
    """
    def eval(self, p, q):
        not_p = luk_not(p)
        out = luk_imp(not_p, q)
        return out
