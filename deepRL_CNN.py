import numpy as np
import matplotlib, cv2
import matplotlib.pyplot as plt
import base64, io, os, time, gym
import IPython, functools
import time
import tensorflow as tf
import tensorflow_probability as tfp
tf.enable_eager_execution()



def choose_action(model, image, single=True):
    # add batch dimension to the observation if only a single example was provided
    image = image.reshape(1,84,84,3)
    logits = model.predict(image) 
    action = tf.random.categorical(logits, num_samples=1)
    action = action.numpy().flatten()
    return action[0] if single else action



### Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

# Compute normalized, discounted, cumulative rewards (i.e., return)
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

### Define the self-driving agent ###
act = tf.keras.activations.relu
Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='valid', activation=act)
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization

# Defines a CNN for the self-driving agent
def CNN_model():
    model = tf.keras.models.Sequential([
        # Convolutional layers
        Conv2D(filters=32, kernel_size=5, strides=2, input_shape=(84,84,3,)),
        Conv2D(filters=48, kernel_size=5, strides=2),
        Conv2D(filters=64, kernel_size=3, strides=2),
        Flatten(),
        # Fully connected layer and output
        Dense(units=512, activation=act),
        Dense(units=3, activation=None)        

    ])
    return model

driving_model = CNN_model()

def compute_loss(logits, actions, rewards): 
   
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)     
    loss = tf.reduce_mean( neg_logprob * rewards ) 

    return loss

## Training parameters and initialization ##

#Learning rate and optimizer '''
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate)
# instantiate driving agent
driving_model = CNN_model()


# instantiate Memory buffer
memory = Memory()

### Training step (forward and backpropagation) ###

max_batch_size = 32
#max_reward = float('-inf') # keep track of the maximum reward acheived during training
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