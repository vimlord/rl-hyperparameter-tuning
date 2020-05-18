
from pbt import *

def execute_training(population,
        max_rounds=1024,
        round_len=8,
        n_threads=4):
    """ Base training behavior given a population with training instructions.
    
    population
            A set of training members to train, complete with initiated
            models and hyperparameters
    max_rounds
            Maximum number of times to stop and attempt exploration
    round_len
            Number of episodes to execute before attempting exploration
    n_threads
            Number of training threads to use. Choose 1 to run sequentially.
    """

    # Evaluate all members
    print('Evaluating initial population')
    for p in population:
        p.evaluate(update=True)

    trainer = Trainer(
            population=population)
    
    print('Beginning training')
    trainer.train(max_rounds=max_rounds,
            round_len=round_len,
            eval_rate=round_len,
            n_threads=n_threads)

    return trainer



