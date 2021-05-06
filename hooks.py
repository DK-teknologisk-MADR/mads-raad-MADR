from detectron2.engine.hooks import HookBase

class save_hook(HookBase):
    def after_step(self):
        if self.trainer.iter % 500== 0:
            print('SAVING TO',f"save_{self.trainer.iter}")
            self.trainer.checkpointer.save(f"model_save_{self.trainer.iter}",iteration = self.trainer.iter)



class StopFakeExc(Exception):
    pass
#implement early stopping
#very hackish, detectron2 can't stop training, but official suggestion is to implement via raised exception.
 # lives in trainer
class early_stop_hook_base(HookBase):
    '''
    IMPORTANT: Stops through raising StopFakeExc so train should be run in try/exc env

    stops early through StopFakeExc(needs try/except outside train)
    meant to be subclassed and implement atleast stopping_criteria

    '''

    def __init__(self,save_name,eval_period):
        self.save_name = save_name
        self.eval_period = eval_period
        self.did_save = True


    def before_step(self):
        self.did_save = False

    def before_stop(self):
        pass

    def stopping_criteria(self):
      '''
      output: (bool) should be true if this is last stop.
      '''
      raise NotImplementedError

    def save(self):
        if not self.did_save:
            self.trainer.checkpointer.save(f"{self.save_name}_{self.trainer.iter}",iteration = self.trainer.iter)

    def _handle_stop(self):
        raise StopFakeExc

    def after_step(self):
        if self.trainer.iter % self.eval_period == 0:
            self.save()
            if self.stopping_criteria():
                self.before_stop()
                self._handle_stop()


class EarlyStopHookBase(HookBase):
    '''
    IMPORTANT: Stops through raising StopFakeExc so train should be run in try/exc env

    stops early through StopFakeExc(needs try/except outside train)
    meant to be subclassed and implement atleast stopping_criteria

    '''

    def __init__(self,save_name):
        self.save_name = save_name
        self.did_save = True


    def before_step(self):
        self.did_save = False

    def before_stop(self):
        pass

    def stopping_criteria(self):
      '''
      output: (bool) should be true if this is last stop.
      '''
      raise NotImplementedError

    def save(self):
        if not self.did_save:
            self.trainer.checkpointer.save(f"{self.save_name}_{self.trainer.iter}",iteration = self.trainer.iter)

    def _handle_stop(self):
        raise StopFakeExc

    def after_step(self):
#        if self.trainer.iter % self.eval_period == 0:
#            self.save()
            if self.stopping_criteria():
                self.before_stop()
                self._handle_stop()

class StopAtIterHook(HookBase):
    def __init__(self,save_name,iter_to_stop):
        super().__init__(save_name)
        self.iter_num = iter_to_stop
        self.iter = 0

    def after_step(self):
        self.iter += 1
        super().after_step()

    def stopping_criteria(self):
        return self.iter_num >= self.iter




#class EarlyStopHook(EarlyStopHookBase):



##early_stop_trainer(DefaultTrainer)

























