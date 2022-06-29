#!/usr/bin/env python

import dqnlearn as dqn
import gym
import numpy
import time
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

    last_time_steps = numpy.ndarray(0)

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


    minibatch_size = 1
    learnStart = 64
    learningRate = 0.001
    discountFactor = 0.99
    memorySize = 100000000

    deepQ = dqn.DeepQ(7, 3, memorySize, discountFactor, learningRate, learnStart)    
    deepQ.initNetworks(learningRate)
    
    # Starts the main training loop 
    for epoch in range(500):
        #get observations
        observation = env.reset()
        #clear the memory            
        deepQ.memory.clear()
        
        while True:
            
            qValues = deepQ.getQValues(observation)
            action = deepQ.selectAction(qValues, explorationRate)
            newObservation, reward, done, info = env.step(action)
            deepQ.addMemory(observation, action, reward, newObservation, done)
            
            if done:
                #if stepCounter <= updateTargetNetwork:
                if(True):
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                    deepQ.memory.clear()
                    break
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)

        observation = newObservation
    env.close()
    