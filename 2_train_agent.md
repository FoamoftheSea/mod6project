# Training the Agent
Instructions for running the parallelized agent training process. Before continuing below, be sure you have already followed the steps in the "Setup" section of the [README file](README.md).

## Parallel Training using Ray
Now that we have seen that our environment will work with ARS, it is time to do some serious training. Running only one car at a time at 15 seconds per episode would take forever to train, so we want to be able to have multiple cars exploring deltas simultaneously and pooling their results together for the update step. To do this, I have altered [the code made available by Mania, Guy, and Recht to reproduce their research on MuJoCo locomotion tasks](https://github.com/modestyachts/ARS) in order to implement the CarEnv instead. Their ARS framework includes functionality on a Ray server. 

All of the code being run to perform this parallel training using ray can be found in the '[ARS_Carla](ARS_Carla/)' folder of this repository. Some noteable changes that I have made to the code include:
 - I have wrapped the observation filter functionality which normalizes the inputs using rolling statistics into a boolean parameter 'state_filter' which defaults to False, since the VGG19 outputs are all on a standard scale between 0-10. Experimenting in training using the filter may be appropriate for future work.
 - I have added learning rate decay, and delta standard deviation (called noise in Skow's code) decay, which can be adjusted by calling --lr_decay and --std_decay parameters. This allows for more exploration early on, and favoring smaller steps over time after some initial progress has been made.
 - I have added functionality which supports initializing the ARS process with a saved policy file, so that training can be resumed at a later time. The user can pass the saved policy file in .csv (weights only) or .npz (weights only or weights plus state filter stats) format into the --policy_file parameter. The code will automatically update the saved policy file 'lin_policy_plus.npz' in the assigned --log_dir location every 10 steps by default, but this can be adjusted by setting the new --log_every parameter on execution. The number of rollouts used in each evaluation step can be set with the new --eval_rollouts.
 - I have added a parameter called --show_cam which takes in an int telling the program how many cameras you would like to display from the group of workers. It defaults to 1, but for long training periods I recommend setting it to 0 in order to save CPU overhead.

### Train a New Policy from Scratch
1. Start CARLA server by running the executable in your CARLA installation.
2. In a **non-administrator** terminal:
   - run `conda activate ars_carla`
   - run `ray start --head`
      - Append parameters to this command that are appropriate for your machine if needed. The CarEnvs take up a lot of RAM, so it may be necessary to set your --object-store-memory and --memory parameters to as high of values as you can.
      - You can call up the list of parameters with 'ray start --help'.
3. Navigate to the ARS_Carla folder in this repository: `cd ARS_Carla`
4. Set this environment variable as recommended by the ARS authors
    - Linux: `export MKL_NUM_THREADS=1`
    - Windows: `$env:MKL_NUM_THREADS=1`
5. Explore list of parameter options for training with `python code/ars_carla.py --help`
6. In another **non-administrator** terminal:
   - run `conda activate ars_carla`
   - Train a policy using the following command (example arguments shown)
```
python code/ars_carla.py --n_iter 1000 --num_deltas 32 --deltas_used 16 --learning_rate 0.02 --lr_decay 0.001 --delta_std 0.03 --std_decay 0.001 --n_workers 4 --show_cam 0 --seconds_per_episode 15 --log_every 10 --eval_rollouts 100 --seed 42 --dir_path .\data\old_logs\2020-12-21_test
```
This will connect to the running Ray cluster, and begin creating workers in the Carla server. The policy will be saved and training details logged every 10 iterations by default into the --dir_path that you set (defaults to ./data if not set).

### Training a Pre-Existing Policy
Follow the same steps above, except include the --policy_file parameter in your call to `code/ars_carla.py` pointing to the .csv or .npz file containing the weights you are going to build on.
   - The default name of a saved policy file is 'lin_policy_plus.npz'.
   - **Note** that this does not automatically recover the current learning rate or delta std from when the training ended, so you will need either recover that from the log file, or start it wherever you want.)

```
python code/ars_carla.py --policy_file ./data/old_logs/2020-12-17_1000steps/lin_policy_plus.npz
```