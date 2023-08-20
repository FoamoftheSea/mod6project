"""
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht

!!!This code has been modified by Nate Cibik in order to work with a Carla
Client Environment.

The CarEnv() Class used is highly based off of that made
by Sentdex in his Deep Q Learning tutorial series using Carla.

Multiple arguments have been added to the argparser at the bottom of this
code to increase the functionality with Carla, as well as to add the ability
to resume training from a pre-existing policy.

Optional Learning rate and Delta Standard Deviation Decay functionality have
been added.
"""

import os
import cv2
import random
import math
import sys
import glob
import time
import logz
import ray
import utils
import optimizers
from policies import *
import socket
from shared_noise import *

from tensorflow.keras.applications import VGG19
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# carla should be installed via conda, or use with a different version
try:
    import carla
except ImportError:
    sys.path.append(glob.glob(r'D:\CARLA_0.9.13\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla

# gpus = tf.config.experimental.list_physical_devices('GPU')
# gpu_memory_limit = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5044)]
# try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], gpu_memory_limit)
# except Exception as e:
#     raise e
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["TF_USE_CUDNN"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# Carla Car Env based on work of Sentdex
class CarEnv:

    def __init__(self,
                 img_width=224,
                 img_height=224,
                 show_cam=False,
                 seconds_per_episode=15,
                 control_type='continuous',
                 car_model='mustang'):
        self.img_width = img_width
        self.img_height = img_height
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.car = self.blueprint_library.filter(car_model)[0]
        self.show_cam = show_cam
        self.control_type = control_type
        self.front_camera = None
        self.actor_list = []
        self.seconds_per_episode = seconds_per_episode
        self.collision_hist = []
        self.steering_cache = []

        if self.control_type == 'continuous':
            self.action_space = np.array(['throttle', 'steer', 'brake'])

    def reset(self):
        self.collision_hist = []
        self.steering_cache = []

        if len(self.actor_list) > 0:
            for actor in self.actor_list:
                actor.destroy()
        self.actor_list = []

        try:
            self.transform = random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(self.car, self.transform)
            self.actor_list.append(self.vehicle)
        except:
            self.reset()

        # Attach RGB Camera
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.img_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.img_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.img_height, self.img_width, 4))
        i3 = i2[:, :, :3]
        if self.show_cam:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action, steps):
        if self.control_type == 'continuous':
            self.vehicle.apply_control(carla.VehicleControl(throttle=np.clip(action[0], 0.0, 1.0),
                                                            steer=np.clip(action[1], -1.0, 1.0),
                                                            brake=np.clip(action[2], 0.0, 1.0)))
        self.steering_cache.append(action[1])
        # elif self.control_type == 'action':
        #     if action == 0:
        #         self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,
        #                                                         steer=-1*self.STEER_AMT))
        #     elif action == 1:
        #        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        #     elif action == 2:
        #         self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,
        #                                                         steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        # Reward System:
        if len(self.collision_hist) != 0:
            # Check to see if on first step (for rough spawns by Carla)
            if steps == 1:
                self.collision_hist = []
                done = False
                reward = 0
            else:
                done = True
                print('Collision!')
                reward = -200
        elif kmh < 60 & kmh > 0.2:
            done = False
            reward = 1  # -1
            # Reward lighter steering when moving
            if np.abs(action[1]) < 0.3:
                reward += 9
            elif 0.5 < np.abs(action[1]) < 0.9:
                reward -= 1
            elif np.abs(action[1]) >= 0.9:
                reward -= 6
        elif kmh <= 0.2:
            done = False
            reward = -10
        else:
            done = False
            reward = 20  # 2
            # Reward lighter steering when moving
            if np.abs(action[1]) < 0.3:
                reward += 20
            # Reduce score for heavy steering
            if 0.5 < np.abs(action[1]) < 0.9:
                reward -= 10
            elif np.abs(action[1]) >= 0.9:
                reward -= 20

        # Penalize consistent and heavily directional steering
        reward -= (np.abs(np.mean(self.steering_cache)) + np.abs(action[1])) * 10 / 2

        if self.episode_start + self.seconds_per_episode < time.time():
            done = True
        if done:
            self.steering_cache = []

        return self.front_camera, reward, done, None


@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 policy_params=None,
                 deltas=None,
                 delta_std=0.02,
                 seconds_per_episode=15,
                 # initial_weights=None,
                 # initial_mean=None,
                 # initial_std=None,
                 show_cam=False,
                 enable_gpu=False):

        # create environment for each worker
        self.env = CarEnv(show_cam=show_cam,
                          seconds_per_episode=seconds_per_episode)
        # self.env.seed(env_seed)
        if enable_gpu:
            self.config = ConfigProto()
            self.config.gpu_options.allow_growth = True
            # if num_workers is not None:
            #    self.config.gpu_options.per_process_gpu_memory_fraction = (1 / num_workers)
            self.session = InteractiveSession(config=self.config)

        # Create base CNN for finding edges
        self.base_model = VGG19(weights='imagenet',
                                include_top=False,
                                input_shape=(self.env.img_height,
                                             self.env.img_width,
                                             3))

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        else:
            raise NotImplementedError

        self.delta_std = delta_std

    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()

    # @ray.remote(num_gpus= .5 / 2)
    def rollout(self, shift=0., state_filter=False):
        """
        At each time-step it subtracts shift from the reward.
        """

        total_reward = 0.
        steps = 0

        state = self.env.reset()
        done = False
        while not done:
            steps += 1
            state = self.env.front_camera.reshape(1, 224, 224, 3) / 255.
            state = self.base_model.predict(state).flatten() / 10
            action = self.policy.act(state, state_filter=state_filter)
            ob, reward, done, _ = self.env.step(action, steps)
            # Clips step reward between -1 and +1
            # reward = max(min(reward, 1), -1)
            total_reward += (reward - shift)

        adjusted_reward = total_reward / steps

        print('Worker saw {} steps'.format(steps))
        return adjusted_reward, steps

    def do_rollouts(self, w_policy, num_rollouts=1, shift=1, delta_std=None,
                    evaluate=False, state_filter=False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)

                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation, we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps = self.rollout(shift=0., state_filter=state_filter)
                rollout_rewards.append(reward)

            else:
                idx, delta = self.deltas.get_delta(w_policy.size)

                if delta_std is None:
                    delta_std = self.delta_std
                delta = (delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps = self.rollout(shift=shift, state_filter=state_filter)

                # compute reward and number of timesteps used for negative perturbation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift=shift, state_filter=state_filter)
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])

        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps": steps}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    def clean_up(self):
        for actor in self.env.actor_list:
            actor.destroy()


class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self,
                 policy_params=None,
                 num_workers=32,
                 num_deltas=320,
                 deltas_used=320,
                 delta_std=0.02,
                 std_decay=0.0,
                 logdir=None,
                 learning_rate=0.01,
                 lr_decay=0.0,
                 shift='constant zero',
                 params=None,
                 seed=123,
                 seconds_per_episode=15,
                 log_every=10,
                 show_cam=1,
                 enable_gpu=False):

        logz.configure_output_dir(logdir)
        logz.save_params(params)

        env = CarEnv()

        # Create base CNN for finding edges
        # base_model = VGG19(weights='imagenet',
        #                     include_top=False,
        #                     input_shape=(env.img_height,
        #                                  env.img_width,
        #                                  3
        #                                 )
        #                     )

        self.timesteps = 0
        self.action_size = env.action_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.delta_std = delta_std
        self.std_decay = std_decay
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.log_every = log_every

        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed=seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.')
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      delta_std=delta_std,
                                      seconds_per_episode=seconds_per_episode,
                                      show_cam=False,
                                      enable_gpu=enable_gpu
                                      # initial_weights=initial_weights,
                                      # initial_mean=initial_mean,
                                      # initial_std=initial_std,
                                      ) for i in range(num_workers - show_cam)]
        # Show the number of desired worker cams
        for i in range(show_cam):
            self.workers.append(Worker.remote(seed + 7 * i,
                                              policy_params=policy_params,
                                              deltas=deltas_id,
                                              delta_std=delta_std,
                                              seconds_per_episode=seconds_per_episode,
                                              show_cam=True,
                                              enable_gpu=enable_gpu,
                                              # initial_weights=initial_weights,
                                              # initial_mean=initial_mean,
                                              # initial_std=initial_std,
                                              ))

        # initialize policy
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError

        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy,
                                        self.learning_rate,
                                        self.lr_decay)
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts=None, evaluate=False, state_filter=False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)

        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                     num_rollouts=num_rollouts,
                                                     shift=self.shift,
                                                     delta_std=self.delta_std,
                                                     state_filter=state_filter,
                                                     evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                     num_rollouts=1,
                                                     shift=self.shift,
                                                     delta_std=self.delta_std,
                                                     state_filter=state_filter,
                                                     evaluate=evaluate)
                           for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], []

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)

        # print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis=1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas

        idx = np.arange(max_rewards.size)[
            max_rewards >= np.percentile(max_rewards, 100 * (1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        top_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        top_rewards /= np.std(top_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(top_rewards[:, 0] - top_rewards[:, 1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size=500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat, rollout_rewards

    def train_step(self, state_filter=False):
        """ 
        Perform one update step of the policy weights.
        """

        g_hat, rewards = self.aggregate_rollouts(state_filter=state_filter)
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        if self.std_decay != 0:
            self.delta_std *= (1 - self.std_decay)
            print('New delta std:', self.delta_std)
        return rewards

    def train(self, num_iter, state_filter=False):

        start = time.time()
        for i in range(num_iter):

            t1 = time.time()
            rewards = self.train_step(state_filter=state_filter)
            t2 = time.time()
            print('total time of one step', t2 - t1)
            print('Iteration', i + 1, 'done')
            print('AverageReward:', np.mean(rewards))
            print('StdRewards:', np.std(rewards))
            print('MaxRewardRollout:', np.max(rewards))
            print('MinRewardRollout:', np.min(rewards))

            # record weights and stats every n iterations
            if (i + 1) % self.log_every == 0:
                rewards = self.aggregate_rollouts(num_rollouts=self.num_deltas,
                                                  evaluate=True)
                # w = ray.get(self.workers[0].get_weights.remote())
                if state_filter:
                    w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                else:
                    w = ray.get(self.workers[0].get_weights.remote())
                np.savez(self.logdir + "/lin_policy_plus", w)

                # print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("Timesteps", self.timesteps)
                logz.log_tabular("LearningRate", self.optimizer.learning_rate)
                logz.log_tabular("DeltaStd", self.delta_std)
                logz.dump_tabular()

            if state_filter:
                t1 = time.time()
                # get statistics from all workers
                for j in range(self.num_workers):
                    self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
                self.policy.observation_filter.stats_increment()

                # make sure master filter buffer is clear
                self.policy.observation_filter.clear_buffer()
                # sync all workers
                filter_id = ray.put(self.policy.observation_filter)
                setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
                # waiting for sync of all workers
                ray.get(setting_filters_ids)

                increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
                # waiting for increment of all workers
                ray.get(increment_filters_ids)
                t2 = time.time()
                print('Time to sync statistics:', t2 - t1)

        return


def run_ars(params):
    dir_path = params['dir_path']

    if not (os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    # Disables TensorFlow GPU use for compatibility reasons.
    # To try and use GPU, set --enable_gpu to True on execution
    if not params['enable_gpu']:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = ConfigProto(device_count={'GPU': 0})
        # config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
    else:
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

    env = CarEnv()
    base_model = VGG19(weights='imagenet',
                       include_top=False,
                       input_shape=(env.img_height,
                                    env.img_width,
                                    3))
    shape = 1
    for num in base_model.output_shape:
        if num is not None:
            shape *= num
    ob_dim = shape  # base_model.input_shape
    ac_dim = env.action_space.shape[0]

    # Set global variable for num workers
    # global worker_count
    # worker_count = params["n_workers"]

    # Get initial weights if directory given. Can be csv or numpy
    if len(params['policy_file']) > 0:
        try:
            initial_policy = np.load(params['policy_file'])
            initial_weights = initial_policy['arr_0']
            print('Found .npz policy file at {}'.format(params['policy_file']))
            print('Loaded policy weights.')
            try:
                initial_mean = initial_policy['arr_1']
                initial_std = initial_policy['arr_2']
                print('Loaded policy stats.')
            except:
                initial_mean = None
                initial_std = None
        except:
            initial_weights = np.genfromtxt(params['policy_file'], delimiter=',')
            print('Found policy file at {}'.format(params['policy_file']))
            print('Loaded weights')
            initial_mean = None
            initial_std = None
    else:
        print('Initializing new policy.')
        initial_weights = None
        initial_mean = None
        initial_std = None

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params = {'type': 'linear',
                     'ob_filter': params['filter'],
                     'ob_dim': ob_dim,
                     'ac_dim': ac_dim,
                     'initial_weights': initial_weights,
                     'initial_mean': initial_mean,
                     'initial_std': initial_std}

    ars = ARSLearner(policy_params=policy_params,
                     num_workers=params['n_workers'],
                     num_deltas=params['num_deltas'],
                     deltas_used=params['deltas_used'],
                     learning_rate=params['learning_rate'],
                     lr_decay=params['lr_decay'],
                     delta_std=params['delta_std'],
                     std_decay=params['std_decay'],
                     logdir=logdir,
                     shift=params['shift'],
                     params=params,
                     seed=params['seed'],
                     seconds_per_episode=params['seconds_per_episode'],
                     show_cam=params['show_cam'],
                     log_every=params['log_every'],
                     enable_gpu=params['enable_gpu']
                     )

    ars.train(params['n_iter'], state_filter=params['state_filter'])

    save_file = '/'.join(params['policy_file'].split('/')[:-1])
    np.savetxt(save_file + '/recent_weights.csv',
               ars.w_policy,
               delimiter=','
               )

    for worker in ars.workers:
        worker.clean_up.remote()

    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', '-n', type=int, default=1000, help="Total number of update steps.")
    parser.add_argument('--num_deltas', '-nd', type=int, default=32,
                        help="Number of deltas to explore between each update step.")
    parser.add_argument('--deltas_used', '-du', type=int, default=16,
                        help="Number of top performing deltas to use in udate step.")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.02, help="Learning rate to start training at.")
    parser.add_argument('--lr_decay', '-lrd', type=float, default=0.0, help="Learning rate decay")
    parser.add_argument('--delta_std', '-std', type=float, default=.03,
                        help="The amount of noise to add to weights during exploration.")
    parser.add_argument('--std_decay', '-stdd', type=float, default=0.0,
                        help="Decay in the amount of noise to add to policy weights.")
    parser.add_argument('--n_workers', '-e', type=int, default=4,
                        help="Number of driver agents to run in parallel. Set based on hardware capability.")
    parser.add_argument('--show_cam', '-sc', type=int, default=1,
                        help="Number of cameras to display to user during training. Set to zero for best performance.")
    parser.add_argument('--policy_file', '-pf', type=str, default='',
                        help="File containing weights for resuming training.")
    parser.add_argument('--seconds_per_episode', '-se', type=int, default=15,
                        help="Maximum number of seconds for each driving episode.")
    parser.add_argument('--state_filter', '-sf', type=bool, default=False,
                        help="Turns on rolling statistics normalization of inputs. Best to leave False.")
    parser.add_argument('--log_every', '-le', type=int, default=10,
                        help="Number of update steps to complete between each logging event.")
    parser.add_argument('--enable_gpu', '-gpu', type=bool, default=False,
                        help="Whether to enable tensorflow gpu access. Leave to False.")

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')

    local_ip = socket.gethostbyname(socket.gethostname())
    ray.init(  # object_store_memory=1.5e+10,
        # _memory=1.5e+10,
        # num_cpus=7,
        # address= local_ip + ':6379',
        # redis_password=''
        # address = '127.0.0.1:6379',
        # local_mode
        # node_ip_address=local_ip
    )

    args = parser.parse_args()
    params = vars(args)

    # with session.as_default():
    run_ars(params)
