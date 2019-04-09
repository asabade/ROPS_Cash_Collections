from abc import ABCMeta, abstractmethod

from pathlib import Path

import pandas as pd


class Experiment(metaclass=ABCMeta):
    """docstring for Experiment"""
    def __init__(self,
                 subject,
                 verb,
                 name,
                 architecture,  # this should have data and model name
                 learner,
                 records: Dict):
        super(Experiment, self).__init__()
        self.architecture = architecture
        self.learner = learner
        if not self.records:
            self.record = {
                'uuid': '234j',
                'subject': self.subject,
                'verb': self.verb,
                'name': self.name,
                'trials': {}
            }


    def _build_path_dirs(self):
        self.experiment_path = Path(self.subject) / self.verb / self.records['uuid']

    def _get_current_place(self): #$..........
        pass


    @classmethod
    def from_records_json(cls, json_path):
        pass
        # def restart_trial.....

    def run_trial(self, hyperparameters: Dict, epochs: int) -> dict:
        self._build_path_dirs()
        retval = {}

        retval['uuid'] = hash(f'{hyperparameters}')
        retval['hyperparameters'] = hyperparameters
        retval['completed'] = 'false'
        self.save_path = self.experiment_path / retval['uuid']

        arch = self.architecture.build(**hyperparameters)
        learned_state = None

        for epoch in range(1, epochs + 1):
            learned = self.learner(model=arch.model,
                                   optimizer=arch.optimizer,
                                   criterion=arch.criterion,
                                   dataloaders=arch.dataloaders,
                                   state=learned_state)

            learned_state = learned.state

            retval['state'] = learned_state

            checkpoint_filepath = self.checkpoint_path / f'{epoch}_{arch.val_loss:.4f}_checkpoint.pth'
            compiled_model_filepath = self.checkpoint_path / f'{epoch}_{arch.val_loss:.4f}_model.pth'

            if epoch == 1 or epoch % 5 == 0:
                
                checkpoint_state = {
                    'epoch': epoch,
                    'model_state_dict': arch.model.state_dict(),
                    'optimizer_state_dict': arch.optimizer.state_dict()
                }

                self.checkpoint(checkpoint_state, self.checkpoint_filepath)
                self.compile(learned.model, self.compiled_model_filepath)

                retval['checkpoint_filepath'] = self.checkpoint_filepath
                retval['compiled_model_filepath'] = self.compiled_model_filepath
            else:
                retval['checkpoint_filepath'] = ''
                retval['compiled_model_filepath'] = ''

            self._save_record()

        retval['loss'] = learned_state['val_loss']
        retval['completed'] = 'true'

        return retval

    def suggestion(self):
        raise NotImplementedError

    def conduct(self, num_trials: int):

        opt = self.suggestion()
        for iteration in num_trials:
            self.hyperparameters_values = opt.ask()
            
            self.hyperparameters = {k: v for k, v in zip(arch.study_name, self.hyperparameters_values)}
            trial = self.run_trial(hyperparameters=hyperparameters)

            self.record['trials'][iteration] = trial
            
            self.loss = trial['loss']
            opt.tell(self.hyperparameters, self.loss)

            self._update_records()


    def compile(self, model, save_path):
        torch.jit.save(model)

    def checkpoint(self, checkpoint_dict, save_path):
        torch.save(checkpoint_dict, save_path)


