# Training Autonomous Vehicles using ARS in Carla
The code in this repository offers a framework for training self-driving cars in the [Carla simulator](https://carla.org/) using [Augmented Random Search (ARS)](https://arxiv.org/pdf/1803.07055.pdf). ARS is an exciting new algorithm in the field of Reinforcement Learning (RL) which has been shown to produce competitive to superior results on benchmark MuJoCo tasks, while offering at least 15x computational efficiency in comparison to other model-free learning methods. This study sought to investigate the application of this algorithm in the context of autonomous vehicles. The computational efficiency of ARS makes it an attractive option for small-scale autonomous driving research (as in the context of this study), but also for scalable exploration into the efficacy various combinations of sensory input and preprocessing techniques for tasks related to autonomous driving.

![The math behind ARS](images/ars_formula_explained.png)

The code in this repository can provide future researchers a framework to investigate the effects of adding various sensors or feature engineering to the learning process, or to use the environment to learn different tasks in the domain of autonomous driving. This study sought to train the agent in the simple task of driving through empty streets without hitting anything, using only the input from a single front-facing camera, as shown below:

![Worker I/O](images/WorkerIO.png)

The results of the study showed that the algorithm was effective at training the agent to perform this basic driving task, and that it may prove useful for tasks related to autonomous driving research in the future.

## Contents
The 'research.ipynb' notebook presents a concise review of the research which made this study possible, offers an intuitive explanation of the math behind the learning algorithm, and provides a basic coding framework of the process which makes it easy to understand the mechanics involved in training the agent, built off of the [ARS framework provided by Colin Skow](https://github.com/colinskow/move37/tree/master/ars). 

The 'train_agent.ipynb' notebook offers guidance on how to get a parallelized training session started using the code in the 'ARS/' folder, and discussion on the possibilities for future development.

The 'ARS/' folder in this repository contains a modified version of the code provided by the authors of the 2018 paper on ARS (Mania, Guy, and Recht) to reproduce their results using the environments in Python's gym module, which employs parallelization through use of the [Ray package](https://docs.ray.io/en/latest/) for Python. Their code has been modified to make use of [Sentdex's CarEnv class](https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/) that he used to train a Deep Q-Learning Network (DQN), which itself has been modified to function in the context of ARS learning. Useful functionality has been added to the ARS code which allows the user to send in a previously trained policy so that training may be resumed at a later time or recovered in the event of an error.

## Training an Agent from Scratch
(The instuctions below apply to Windows 10, but should be similar for other OS. For more discussion of the steps taken below, see the 'train_agent.ipynb' notebook in this repository.)

To see the learning algorithm in action on the task of driving through the empty streets of Carla, you will first need to have Carla installed, as well as Ray, Tensorflow, and OpenCV packages for Python.

Then, once you have the dependencies installed, from a *non-administrator* terminal (Windows Powershell works), start a Ray cluster by typing the following command (note that you may need to adjust parameters for this command to suit the hardware of your machine, which you can inspect by calling 'ray start --help'. Running the command with no additional parameters will automatically set the number of CPUs, GPUs, and memory reserves based on your machine):

```
ray start --head
```

Once the cluster is successfully started, start Carla, then open a separate non-administrator terminal, navigate to the 'ARS/' folder in this repository, and enter the following command (parameters are shown being set for demonstration. Note that the --dir_path you set will contain the log file and policy file related to your training session, and will be created if it does not exist):

```
python code/ars.py --n_iter 1000 --n_workers 4 --num_deltas 32 --deltas_used 16 --learning_rate 0.02 --lr_decay 0.001 --delta_std 0.03 --std_decay 0.001 --show_cam 1 --seed 42 --dir_path ./data/old_logs/2020-12-17_1000steps
```

This will connect to the running Ray cluster, and begin creating workers in the Carla server. The policy will be saved and training details logged every 10 iterations by default into the --dir_path that you set (defaults to ./data if not set).

## Training a Pre-Existing Policy
As above, you will need to start the Ray cluster and Carla server, then run a similar command as above, except you will include the --policy_file parameter, which points to the .csv or .npz file which contains the weights you are going to build on. (The default name of a saved policy file is 'lin_policy_plus.npz'. Note that this does not automatically recover the current learning rate or delta std from when the training ended, so you will need either recover that from the log file, or start it wherever you want.)

```
python code/ars.py --policy_file ./data/old_logs/2020-12-17_1000steps/lin_policy_plus.npz
```