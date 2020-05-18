
import torch
from torch import nn
from torch import optim

import random

from procedure import TrainingProcedure

class PPOTrainingProcedure(TrainingProcedure):
    def __init__(self,
            env,
            use_reward_normalization=False,
            use_gae=True):

        super().__init__(env)
        
        self.use_reward_normalization = use_reward_normalization
        self.use_gae = use_gae

        self.data = None

    def do_single_training_round(self, agent, config):
        # Do the training
        trace = self.run_training_episode(agent=agent, **config)
        self.do_training_step(agent=agent, trace=trace,
                optimizer=config.optimizer, **config)

    def run_training_episode(self, agent, episode_length, **args):
        env = self.env
        
        state = env.reset()

        S0, A, S1, P, R, D = [[] for _ in range(6)]

        for t in range(episode_length):
            S0.append(state)
            
            a, p = agent.step(state)
            A.append(a)
            P.append(p)

            s1, r, d = env.step(a)[:3]
            S1.append(s1)
            R.append(r)
            D.append(d)

            # Reset on episode end
            if d > 0:
                state = env.reset()
            else:
                state = S1[t]
        
        S0 = torch.stack(S0).detach()
        A = torch.stack(A).detach()
        S1 = torch.stack(S1).detach()
        P = torch.stack(P).detach()
        R = torch.Tensor(R).cuda()
        D = torch.Tensor(D).cuda()

        if self.use_reward_normalization:
            R = ((R - R.mean()) / (R.std() + 1e-6)).detach()

        return S0, A, S1, P, R, D

    def do_training_step(self, 
            agent,
            trace,
            optimizer,
            L,
            batch_size,
            episode_length,
            gamma,
            lmbda,
            eps,
            n_updates_per_step,
            **args):

        # Memory
        S0, A, S1, P, R, D = trace

        log_prob_old = P.log()
    
        # Compute evaluations of V at each time
        V0 = agent.value(S0).squeeze()
        V1 = agent.value(S1).squeeze()

        # Compute true V0
        V_after = (R + gamma * V1 * (1-D)).detach()
        
        # Compute advantage
        if self.use_gae:
            Adv = torch.zeros((A.size()[0]),).cuda()

            # Mask determines whether or not to permit propagation of future Q/V values
            # This allows separation of episodes.
            mask = 1-D

            adv = 0
            for t in range(A.size()[0]-1, -1, -1):
                # Current mask
                m = mask[t]

                # Propagate
                delta = R[t] + gamma * V1[t] * m - V0[t]
                adv = Adv[t] = delta + gamma * lmbda * adv * m

            # Detach to prevent gradient computations
            Adv = Adv.detach()
        else:
            # Advantage is how much better the value is in the future versus in the past
            Adv = (V_after - V0).detach()

        del V0
        del V1

        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-6)

        #print('Running training steps')

        for ep in range(n_updates_per_step):
            idxs = list(range(A.size()[0]))
            random.shuffle(idxs) 

            # Shuffle the arrays
            S0 = S0[idxs]
            S1 = S1[idxs]
            A = A[idxs]
            Adv = Adv[idxs]
            V_after = V_after[idxs]
            log_prob_old = log_prob_old[idxs]
            
            for i in range(0, episode_length, batch_size): 
                # Reset gradients
                optimizer.zero_grad()
                j = i + batch_size

                # Compute the predicted value, log probabilities of choosing actions, and the entropy
                V_before, log_prob, entropy = agent.reevaluate(S0[i:j], A[i:j])

                # We wish to maximize entropy
                ent_loss = -entropy

                # The reward should more closely match reality
                V_loss = nn.MSELoss()(V_before, V_after[i:j])

                # Compute PPO loss
                ratio = torch.exp(log_prob - log_prob_old[i:j])

                arg1 = ratio * Adv[i:j]
                arg2 = torch.clamp(ratio, 1-eps, 1+eps)
                
                ppo_loss = -torch.min(arg1, arg2)

                # Combined loss
                loss = ppo_loss.mean() + 0.5 * V_loss + 0.01 * ent_loss.mean()

                # Apply gradients
                loss.backward()
                optimizer.step()

