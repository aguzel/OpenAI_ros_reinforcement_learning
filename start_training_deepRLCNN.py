#!/usr/bin/env python

import gym
import numpy
import time
import qlearn_basic as qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import deepRL_CNN as deepRLCNN
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


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

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time() 
    bridge_object = CvBridge()

    for i_episode in range(500):
        cumulated_reward = 0 
        
        # Restart the environment
        observation = env.reset()
        deepRLCNN.memory.clear()
        
        
        for x in range(nsteps):

            cv_image = bridge_object.imgmsg_to_cv2(env.get_camera_rgb_image_raw() , desired_encoding="bgr8")            
            cv2.imshow("Image window", cv_image)
            cv2.waitKey(1)
            
            #image processing - adapted from baseline VM writtent by Dr. Altahhan
            model_image_size = (84,84)
            img = cv2.resize(cv_image, model_image_size, interpolation = cv2.INTER_CUBIC)
            img = img.astype(np.float32)
            img /= 255.
            img = img.reshape(84,84,3)
            observation = img

            action = deepRLCNN.choose_action(deepRLCNN.driving_model, observation) 

            _, reward, done, _ = env.step(action)
            cumulated_reward += reward

            
            # add to memory
            deepRLCNN.memory.add_to_memory(observation, action, reward)
            
            # Make the algorithm learn based on the results
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# Cumulative reward that action gave=>" + str(cumulated_reward))
            
            # is the episode over? did you crash or do so well that you're done?
            if done:
                # determine total reward and keep a record of this
                total_reward = sum(deepRLCNN.memory.rewards)
                
                rospy.logwarn("# total_reward that actions gave=>" + str(total_reward))
                
                batch_size = min(len(deepRLCNN.memory), deepRLCNN.max_batch_size)
                #i = np.random.choice(len(deepRLCNN.memory), batch_size, replace=False)
                deepRLCNN.train_step(deepRLCNN.driving_model, deepRLCNN.compute_loss, deepRLCNN.optimizer, 
                                observations=np.array(deepRLCNN.memory.observations),
                                actions=np.array(deepRLCNN.memory.actions),
                                discounted_rewards = deepRLCNN.discount_rewards(deepRLCNN.memory.rewards))     
                       
                # reset the memory
                deepRLCNN.memory.clear()
                break
    env.close()
        