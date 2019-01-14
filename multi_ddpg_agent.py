import numpy as np
import random
import copy
from collections import namedtuple, deque
import time
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    '''
    MultiAgent DDPG implementation
    '''

    def __init__(self, 
                 number_of_agents, 
                 state_size, 
                 action_size, 
                 rand_seed,
                 BUFFER_SIZE = int(1e5),
                 BATCH_SIZE  = 250):
        '''
        Creates a new instance of MultiAgent DDPG agent
        
        agent_qty: indicates the number of agents.
        state_size: size of the observation space
        action_size: size of the actions space
        '''    

        self.number_of_agents = number_of_agents
        self.batch_size = BATCH_SIZE 
        self.buffer_size = BUFFER_SIZE
       
        # Creating Replay Buffer for multiple agents
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, rand_seed, self.number_of_agents)
     
        self.action_size = action_size
        self.state_size = state_size 
        print('r seed is: ', rand_seed)
        self.agents = [DDPGAgent(i, self.state_size,
                               self.action_size,
                               rand_seed,
                               self)
                         for i in range(self.number_of_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        """ Add into memory the experience. It will include all agents information """

        self.memory.add(states, actions, rewards, next_states, dones)

        # For each agent perfoms a step 
        for agent in self.agents:
            agent.step()

    def act(self, states, add_noise=True):
        """ Returns the action to take """
        na_rtn = np.zeros([self.number_of_agents, self.action_size])
        for agent in self.agents:
            idx = agent.agent_id
            na_rtn[idx, :] = agent.act(states[idx], add_noise)
        return na_rtn

    def save_weights(self):
        """ Save agents weights"""
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(agent.agent_id+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(agent.agent_id+1))
    
    def reset(self):
        """ Just reset all agents """
        for agent in self.agents:
            agent.reset()

    def __len__(self):
        return self.number_of_agents

    def __getitem__(self, key):
        return self.agents[key]
        
class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent_id, 
                 state_size, 
                 action_size, 
                 rand_seed, 
                 meta_agent,
                 BUFFER_SIZE = int(1e5),
                 BATCH_SIZE = 250, 
                 GAMMA = 0.99,
                 TAU = 1e-3,
                 LR_ACTOR = 1e-4,
                 LR_CRITIC = 1e-3,
                 UPDATE_EVERY = 2, 
                 WEIGHT_DECAY = 0):
        """
        Initialize an Agent object.
        
        Params
        ======
            state_size (int) : dimension of each state
            action_size (int): dimension of each action
            rand_seed (int): random seed
            BUFFER_SIZE (int): replay buffer size
            BATCH_SIZE (int): minibatch size
            GAMMA (float): discount factor
            TAU (float): for soft update of target parameters
            LR_ACTOR (float): learning rate for critic 
            LR_CRITIC (float): learning rate for critic
            WEIGHT_DECAY (float): L2 weight decay
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.update_every = UPDATE_EVERY
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, rand_seed).to(device)
        self.actor_target = Actor(state_size, action_size, rand_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), \
                                            lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, meta_agent.number_of_agents, rand_seed).to(device)
        self.critic_target = Critic(state_size, action_size, meta_agent.number_of_agents, rand_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),\
                                            lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, rand_seed)

        # Time step
        self.timestep = 0
        
        # Replay memory
        self.memory = meta_agent.memory
        
    def step(self):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #for s, a, r, ns, d in zip(state, action, reward, next_state, done):
        #    self.memory.add(s, a, r, ns, d)
            
        self.timestep = (self.timestep + 1) % self.update_every
        
        if self.timestep == 0:
            # if enough samples are available in memory
        
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
                


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        (states_list, actions_list, rewards, next_states_list, dones) = experiences
        
        list_all_next_actions = []
        for states in states_list:
            list_all_next_actions.append(self.actor_target(states))
        
        # Convert the experiences into Torch tensors
        all_next_actions = torch.cat(list_all_next_actions, dim=1).to(device)
        all_next_states = torch.cat(next_states_list, dim=1).to(device)
        all_states = torch.cat(states_list, dim=1).to(device)
        all_actions = torch.cat(actions_list, dim=1).to(device)
        
        # ---------------------------- update critic ---------------------------- #
        
        # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        
        actions_pred = []
        for states in states_list:
            actions_pred.append(self.actor_local(states))
        
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
        
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, number_of_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.number_of_agents = number_of_agents
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        list_of_states = list()
        list_of_actions = list()
        list_of_next_states = list()

        
        for i in range(self.number_of_agents):
            index = np.array([i])
            states = torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device)
            
            list_of_states.append(states)
            list_of_actions.append(actions)
            list_of_next_states.append(next_states)
            
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)        
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (list_of_states, list_of_actions, rewards, list_of_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
