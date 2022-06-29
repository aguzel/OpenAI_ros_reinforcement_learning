#!/usr/bin/env python

import gym
import numpy as np
import time
import qlearn_basic as qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import deepRL


if __name__ == '__main__':

    rospy.init_node('example_turtlebot2_maze_qlearn',
                    anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot2/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    #mitigate any random variables from each simulation
    env.seed(1)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot2_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot2/alpha")
    Epsilon = rospy.get_param("/turtlebot2/epsilon")
    Gamma = rospy.get_param("/turtlebot2/gamma")
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot2/nepisodes")
    nsteps = rospy.get_param("/turtlebot2/nsteps")

    running_step = rospy.get_param("/turtlebot2/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()

    # Starts the main training loop 
    for i_episode in range(500):
        # Restart the environment
        observation = env.reset()        
        #initialize memory
        deepRL.memory.clear()
        while True:
            # using our observation, choose an action and take it in the environment
            action = deepRL.choose_action(deepRL.turtlebot_model, observation)
            next_observation, reward, done, info = env.step(action)         
            #add to memory
            deepRL.memory.add_to_memory(observation, action, reward)

            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))


            if done:    
                # initiate training
                g = deepRL.train_step(deepRL.turtlebot_model, deepRL.compute_loss, deepRL.optimizer, 
                       observations=np.vstack(deepRL.memory.observations),
                       actions=deepRL.np.array(deepRL.memory.actions),
                       discounted_rewards = deepRL.discount_rewards(deepRL.memory.rewards))                       
          
                # reset the memory
                deepRL.memory.clear()
                break
            # update our observatons
        observation = next_observation
    env.close()
