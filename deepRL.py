import numpy as np
import matplotlib, cv2
import matplotlib.pyplot as plt
import base64, io, os, time, gym
import IPython, functools
import time
import tensorflow as tf
tf.enable_eager_execution()

def choose_action(model, observation, single=True):
    # add batch dimension to the observation if only a single example was provided
    observation = np.expand_dims(observation, axis=0) if single else observation

    logits = model.predict(observation) 

    action = tf.random.categorical(logits, num_samples=1)

    action = action.numpy().flatten()

    return action[0] if single else action

### Define the agent ###

# Defines a feed-forward neural network

def create_turtlebot2_model():
    model = tf.keras.models.Sequential([
        # First Dense layer
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),        
        
        # Think about the space the agent needs to act in!
        tf.keras.layers.Dense(units=3, activation=None) 
       
    ])
    return model

turtlebot_model = create_turtlebot2_model()


### Agent Memory ###

class Memory:
    def __init__(self): 
        self.clear()

  # Resets/restarts the memory buffer
    def clear(self): 
        self.observations = []
        self.actions = []
        self.rewards = []

  # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward): 
        self.observations.append(new_observation)
        
        self.actions.append(new_action)        
        self.rewards.append(new_reward) 
        

    def __len__(self):
        return len(self.actions)

# Instantiate a single Memory buffer
memory = Memory()


### Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

# Compute normalized, discounted, cumulative rewards
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, gamma=0.95): 
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
      
    return normalize(discounted_rewards)

### Loss function ###

# Arguments:
#   logits: network's predictions for actions to take
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
# Returns:
#   loss
def compute_loss(logits, actions, rewards): 
   
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)     

    loss = tf.reduce_mean( neg_logprob * rewards ) 
    print("LOSS of Network : " + str(loss))

    return loss

def train_step(model, loss_function, optimizer, observations, actions, discounted_rewards, custom_fwd_fn=None):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
        if custom_fwd_fn is not None:
            prediction = custom_fwd_fn(observations)
        else: 
            prediction = model(observations)
  
        loss = loss_function(prediction, actions, discounted_rewards) 

    grads = tape.gradient(loss, model.trainable_variables) 
    grads, _ = tf.clip_by_global_norm(grads, 2) 
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

## Training parameters ##
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)



