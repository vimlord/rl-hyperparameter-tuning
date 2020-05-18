
from collections.abc import Mapping

class TrainingProcedure:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape
        self.n_actions = env.action_space.n

    def do_training(self, agent, config):
        """ Perform training on a single agent for
        several rounds based on the given configuration.

        agent
                The agent to optimize
        config
                The set of hyperparameters to use for training
        """
        for _ in range(config['n_training_steps']):
            self.do_single_training_round(agent, config)
    
    def do_single_training_round(self, agent, config):
        raise NotImplementedError

class Configuration(Mapping):
    def __init__(self, generator, **args):
        self.args = args
        self.generator = generator

    def __getitem__(self, key):
        return self.args[key]

    def __iter__(self):
        for k in self.args:
            yield k
    
    def __len__(self):
        return len(self.args)

    def __setitem__(self, key, value):
        self.args[key] = value

    def __getattr__(self, key):
        return self[key]

    def copy(self):
        return Configuration(**self, generator=self.generator)

    def clone(self):
        return self.copy()

    def mutate(self, **args):
        return self.generator.mutate(self, **args)

class ConfigurationGenerator:
    def __init__(self, **args):
        self.args = args

    def generate(self, **args):
        return Configuration(**{
                k: self[k](**args) for k in self.args},
                generator=self)

    def siphon(self, **args):
        while True:
            yield self.generate(**args)

    def range(self, ct, **args):
        i = 0
        for x in self.siphon(**args):
            yield x

            i += 1
            if i >= ct: return

    def __getitem__(self, key):
        return self.args[key]

    def mutate(self, config, **args):
        return Configuration(**{
                k: self[k](config=config, value=config[k], **args)
                for k in config}, generator=self)

    def copy(self, config):
        return {k: config[k] for k in config}

