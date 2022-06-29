This projects explores the challenge of autonomous mobile robot navigation in complex environments. The objective of autonomous mobile robot navigation is to reach goal position and return back to original position, without colliding with obstacles. Between two objectives, there is another task which is robot should manipulate the object on the goal position, before returning the starting position of task one.

Instead of path planning and SLAM algorithms, deep reinforcement learning is applied in this project. There are two different reinforcement learning methods are applied during start position to goal and goal to start position. Both methods are designed with neural network architecture (deep reinforcement learning).

1st Method : Deep Reinforcement Learning 1 (Policy Gradient) for navigation 1

2nd Method : Deep Reinforcement Learning 2 (DQN) for navigation 2

In this project Gazebo is used as simulator, since it best supports the requirements for training the navigation robot agent. It provides physics engine and sensors simulation at faster than real world physical robot. OpenAI Robot Operating System (OpenAI ROS) is also used in this project. OpenAI ROS interfaces directly with Gazebo, without necessitating any changes to software in order to run in simulation as opposed to the physical world, and it provides wide range reinforcement Learning libraries that allow to train turtlebot on tasks. Since creating this simulation ecosystem for the project is time consuming, a virtual machine is provided by the instructor is used instead of creating from sctrach.Instead of VM ready world,a new world is created for navigation task.

Hyper parameter tunning and reward function design are studied,and results are compared in terms of total reward after each step.

![image](https://user-images.githubusercontent.com/46696280/176344382-30e2c9b9-a365-4637-9027-f05ddc545e8c.png)


![image](https://user-images.githubusercontent.com/46696280/176344425-8cf2f9bc-d492-4b8f-8644-54a70dc9d80a.png)
