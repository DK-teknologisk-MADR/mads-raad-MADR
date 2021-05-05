from math import log,floor,exp
from numpy.random import standard_cauchy
import numpy as np
from scipy.special import lambertw

class pruner_base():
    '''
    Requirement of any pruner:
    input: Max_res
    attributes: participants (max)
    function: All public functions below
    '''
    def report_and_get_next_trial(self,reported_val):
        '''
        MUST return
        input: reported_val score to book-keep for current running trial.
        output: 3-Tuple of
        (int)next_trial to run
        list(int) list of trials that should be pruned
        (bool) whether we are finished or not.
        '''
        raise NotImplementedError

class SHA(pruner_base):
    #before named SHA_BOOK

    def __init__(self, max_res = None , factor = 3,rungs= None,topK=1):
        assert max_res is not None or rungs is not None , "either rungs or max_res should be specified"
        if rungs is not None:
            if max_res is not None:
                print("both max_res and rungs has been specified. Will ignore rungs")
            else:
                max_res = pow(factor,rungs)*rungs
        else:
            rungs = floor(0.0001 + np.real(lambertw(log(factor) * max_res) / log(factor)))
        self.max_res = max_res
        self.rungs = rungs
        self.factor=factor
        self.MINIMUM = -99999999999
        self.rungs = floor(0.0001+np.real(lambertw(log(factor)*self.max_res)/log(factor)))
        self.participants = floor(pow(factor,self.rungs)) #maybe not +1
        #     print("number of rungs are", self.rungs, "and number of participants are ", self.participants)
        #----construct rungs
        self.rung_results = [[[j,-99999999999] for j in range(pow(factor,self.rungs-i))] for i in range(0,self.rungs+1)]
        self.rung_cur = 0
        self.trial_in_rung_cur = 0
        self.max_trials_in_rung_cur = len(self.rung_results[0])
        self.done = False
        self.rung_to_stop = floor(self.rungs-log(topK,factor)+0.0001) #TODO: Not implemented yet, use it.

    def get_cur_res(self):
        return int(pow(self.factor,self.rung_cur))

    def report_and_get_next_trial(self,reported_val):
        if self.done:
            raise ValueError
        self.rung_results[self.rung_cur][self.trial_in_rung_cur][1] = reported_val
        self.trial_in_rung_cur+=1
        pruned = []
        if self.trial_in_rung_cur == self.max_trials_in_rung_cur:
            next_rung = self.rung_cur +1
            self.next_rung_size = self.max_trials_in_rung_cur // self.factor
            self.rung_results[self.rung_cur].sort(reverse = True, key=lambda tup: tup[1])
        #        print("done with rung, rung cur is",self.rung_cur)
        #        print(self.rung_results[self.rung_cur])
            for winner_nr in range(self.next_rung_size):
                self.rung_results[next_rung][winner_nr] = self.rung_results[self.rung_cur][winner_nr]
            pruned = self.rung_results[self.rung_cur][self.next_rung_size :]
            pruned = [tup[0] for tup in pruned]
#        print(pruned)
            self.rung_cur = next_rung
            self.max_trials_in_rung_cur = self.next_rung_size
            self.trial_in_rung_cur = 0
        if self.rung_cur>=self.rung_to_stop:
            self.done = True
        self.next_trial = self.rung_results[self.rung_cur][self.trial_in_rung_cur][0]
        print("CURRENT RUNG IS ",self.rung_cur, "AND CURRENT TRIAL IS", self.trial_in_rung_cur,"PRUNED IS",pruned)
        return self.next_trial, pruned , self.done


    def get_best_models(self):
        return self.rung_results[self.next_rung]

#TEST FUNCTION
    def _start(self):
        done = False
        while not done:
            next_trial,pruned,done = self.report_and_get_next_trial(floor(abs(standard_cauchy())))
