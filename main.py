#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Copyright 2021 Arthur Müller and Vishal Rangras
This file is part of LemgoRL.

LemgoRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LemgoRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LemgoRL.  If not, see <http://www.gnu.org/licenses/>.

----------------
Main function for training and evaluating deep reinforcement learning agents
for traffic light control in sumo environments
@author: Arthur Müller (Fraunhofer IOSB-INA in Lemgo)
@email: arthur.mueller@iosb-ina.fraunhofer.de
"""

import argparse
import importlib
import logging
from config.excelparser_parameters import parse_algo_parameters
from utils import CustomPlot
from greedy import LongestQueueFirstPolicyLemgoWvc
import numpy as np
from pathlib import Path
import time

logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.DEBUG)

BASE_PATH = '.'
RESULTS_PATH = './results/'
QUEUE_RESULTS_PATH = '/queue-results/'
WAIT_RESULTS_PATH = '/wait-results/'
SPEED_RESULTS_PATH = '/speed-results/'
PHASE_WAVE_RESULT_PATH = '/phase-wave-results/'
PEDESTRIAN_RESULT_PATH = '/pedestrian-wait-results/'


def parse_args():
    """
    Parse the arguments from shell.
    :return: args object
    """
    default_base_dir = BASE_PATH

    parser = argparse.ArgumentParser()

    parser.add_argument('--base-dir',
                        type=str,
                        required=False,
                        default=default_base_dir,
                        help="BASE_DIR for agents, envs etc(Git-Repo)",)
    parser.add_argument('--env',
                        type=str,
                        required=False,
                        default='owl322',
                        help="env to train or evaluate an agent. default=one_intersection",
                        choices=['owl322'])
    parser.add_argument('--gui',
                        type=bool,
                        required=False,
                        default=False,
                        help="Visualize training or evaluate with sumo-gui",)
    parser.add_argument('--controlled-nodes',
                        type=str, default='OWL322',
                        required=False,
                        help="LemgoRL supports only OWL322 as controlled node at present.")
    parser.add_argument('--vc-mode',
                        type=bool,
                        required=False,
                        default=True,
                        help="LemgoRL supports processing with virtual controller in loop.")
    parser.add_argument('--host',
                        type=str,
                        required=False,
                        default='localhost',
                        help="Host address for Lisa+ Virtual Controller server")
    parser.add_argument('--port',
                        type=int,
                        required=False,
                        default=9081,
                        help="Port for Lisa+ Virtual Controller server")
    parser.add_argument('--server-path',
                        type=str,
                        required=False,
                        default='./lisa_vissim_addon/OmlFgServer.jar',
                        help="Location for Lisa+ Virtual Controller server path")
    parser.add_argument('--data-dir',
                        type=str,
                        required=False,
                        default='./virtual_controller/VISSIM_OWL322/',
                        help="Data directory for Lisa+ Virtual Controller program files")
    parser.add_argument('--lisa-cfg-file',
                        type=str,
                        required=False,
                        default='./lisa_interface/lisa_conf/lisa_config.yaml',
                        help="Path to Lisa+ Virtual Controller configuration file.")
    parser.add_argument('--sumo-connector',
                        type=str,
                        required=False,
                        default='libsumo',
                        help="Choose sumo-connector from traci and libsumo. Default is traci.",
                        choices=['traci', 'libsumo'])

    subparsers = parser.add_subparsers(dest='modus',
                                       help="train, evaluate or visualize")

    subp_evaluate = subparsers.add_parser('evaluate', help="evaluate agents")

    subp_evaluate.add_argument('--algo',
                               type=str, default='greedy',
                               required=False,
                               help="LemgoRL currently supports greedy policy as default algorithm.")
    subp_evaluate.add_argument('--checkpoint-nr', type=int, required=False, default=0,
                               help="Which checkpoint-nr number from trained agents to use.",)
    subp_evaluate.add_argument('--mdp_sumo_id', type=str, required=False, default='last',
                               help="ID to choose the parameters for MDP and "
                                    "sumo in corresponding config-file. default=last")
    subp_evaluate.add_argument('--episodes', type=int, required=False, default=5,
                               help="Number of episodes for which we want "
                                    "to run the simulation for evaluation. default = 5")
    subp_evaluate.add_argument('--verbose', type=bool, required=False, default=False,
                               help="Plots graph of each episode separately. default = False")
    subp_evaluate.add_argument('--debug_metrics', type=bool, required=False, default=False,
                               help="Print metric values inside environment. default=False")
    subp_evaluate.add_argument('--ci', type=bool, required=False, default=False,
                               help="Plots Confidence Interval of the evaluation graphs. default=False")

    subp_vis = subparsers.add_parser('visualize', help="visualizes an agent's policy in a sumo environment")

    subp_vis.add_argument('--checkpoint-nr', type=int, required=False, default=0,
                          help="Checkpoint Number for the agent to visualize. Not necessary for rule-based agent.", )
    subp_vis.add_argument('--algo', type=str, required=False, default='greedy',
                          choices=['greedy'],
                          help="LemgoRL currently supports greedy policy as default algorithm.")
    subp_vis.add_argument('--mdp_sumo_id', type=str, required=False, default='last',
                          help="ID to choose the parameters for MDP and sumo in "
                               "corresponding config-file. default=last")
    subp_vis.add_argument('--print_info', type=bool, required=False, default=False,
                          help="Print observation, reward, done, info from environment. default=False")
    subp_vis.add_argument('--debug_metrics', type=bool, required=False, default=False,
                          help="Print metric values inside environment. default=False")

    args = parser.parse_args()
    if not args.modus:
        parser.print_help()
        exit(1)
    return args


def parse_args_and_init_agent_env(args):
    # init env
    mdp_sumo_param_path = args.base_dir + r'/config/mdp_sumo_parameters.xlsx'
    sumo_dict = parse_algo_parameters(excel_config_path=mdp_sumo_param_path,
                                      param_group='sumo',
                                      desired_id=args.mdp_sumo_id)
    mdp_dict = parse_algo_parameters(excel_config_path=mdp_sumo_param_path,
                                     param_group='mdp',
                                     desired_id=args.mdp_sumo_id)
    sumo_env_config_dir = args.base_dir + r'/envs/sumo_files_' + args.env

    if args.vc_mode:
        env_module = importlib.import_module('envs.' + args.env + '_wvc')
    else:
        env_module = importlib.import_module('envs.' + args.env)

    # create env_config without gui option
    env_config = {
        'env_config': {
            'sumo_dict': sumo_dict,
            'mdp_dict': mdp_dict,
            'sumo_env_config_dir': sumo_env_config_dir,
        }
    }
    env_config['env_config']['env'] = args.env
    env_config['env_config']['controlled_nodes'] = [args.controlled_nodes]
    env_config['env_config']['sumo_connector'] = args.sumo_connector
    if args.modus == 'evaluate' or args.modus == 'visualize':
        env_config['env_config']['skip_worker'] = True
    else:
        env_config['env_config']['skip_worker'] = False

    if args.debug_metrics:
        env_config['env_config']['print_metrics'] = True
    else:
        env_config['env_config']['print_metrics'] = False

    if args.vc_mode:
        env_config['env_config']['lisa_params'] = {
            'host': args.host,
            'port': args.port,
            'server_path': args.server_path,
            'data_dir': args.data_dir,
            'lisa_cfg_file': args.lisa_cfg_file
        }

    if len(args.controlled_nodes) == 2:
        state_space_config = {
            'queue_start_ind': 48,
            'wave_start_ind': 24
        }
    else:
        state_space_config = {
            'queue_start_ind': 24,
            'wave_start_ind': 12
        }
    env_config['env_config']['state_space_config'] = state_space_config
    agent = LongestQueueFirstPolicyLemgoWvc(env_config['env_config'])
    env_config['env_config']['gui'] = args.gui
    return agent, env_config


def evaluate(args):
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    QUEUE_PATH = RESULTS_PATH + timestamp + QUEUE_RESULTS_PATH
    WAIT_PATH = RESULTS_PATH + timestamp + WAIT_RESULTS_PATH
    SPEED_PATH = RESULTS_PATH + timestamp + SPEED_RESULTS_PATH
    PHASE_WAVE_PATH = RESULTS_PATH + timestamp + PHASE_WAVE_RESULT_PATH
    PEDESTRIAN_PATH = RESULTS_PATH + timestamp + PEDESTRIAN_RESULT_PATH
    Path(QUEUE_PATH).mkdir(parents=True, exist_ok=True)
    Path(WAIT_PATH).mkdir(parents=True, exist_ok=True)
    Path(SPEED_PATH).mkdir(parents=True, exist_ok=True)
    Path(PHASE_WAVE_PATH).mkdir(parents=True, exist_ok=True)
    Path(PEDESTRIAN_PATH).mkdir(parents=True, exist_ok=True)

    algo_list = [args.algo]
    checkpoint_list = [args.checkpoint_nr]

    queue_length_episodes_algos = []
    wait_time_episodes_algos = []
    speed_episodes_algos = []
    pedestrian_episodes_algos = []
    std_err_que_algos = []
    std_err_wait_algos = []
    std_err_pedestrian_algos = []

    if args.vc_mode:
        timesteps = 5400
    else:
        timesteps = 1080  # 5400/5

    max_episode = args.episodes
    t2 = np.arange(1., max_episode + 1)

    for algo, checkpoint_nr in zip(algo_list, checkpoint_list):
        args_dict = vars(args)
        args_dict['algo'] = algo
        args_dict['checkpoint_nr'] = checkpoint_nr

        agent, env_config = parse_args_and_init_agent_env(args)

        if args.vc_mode:
            env_module = importlib.import_module('envs.' + args.env + '_wvc')
        else:
            env_module = importlib.import_module('envs.' + args.env)
        env = env_module.TrafficSimulation(env_config['env_config'])

        curr_episode = 1

        queue_length_episodes = []
        wait_time_episodes = []
        speed_episodes = []
        pedestrian_wait_episodes = []

        queue_length_episodes_std = []
        wait_time_episodes_std = []
        speed_episodes_std = []
        pedestrian_wait_episodes_std = []

        CustomPlot.plot_figure()

        while curr_episode <= max_episode:
            obs = env.reset()
            done = False
            step = 1
            queue_length_steps = []
            wait_time_steps = []
            speed_steps = []
            pedestrian_wait_steps = []
            while not done:
                action = agent.compute_action(obs)
                obs, reward, done, info = env.step(action)

                queue_length_nodes_mean = env.getQueueLengthsNodeMean()
                queue_length_steps.append(queue_length_nodes_mean)

                wait_time_nodes_mean = env.getWaitTimeNodeMean()
                wait_time_steps.append(wait_time_nodes_mean)

                speed_nodes_mean = env.getSpeedNodeMean()
                speed_steps.append(speed_nodes_mean)

                pedestrian_wait_nodes_mean = env.getPedestrianWaitNodeMean()
                pedestrian_wait_steps.append(pedestrian_wait_nodes_mean)

                step += 1

            queue_length_steps_mean = np.mean(queue_length_steps)
            queue_length_episodes.append(queue_length_steps_mean)

            wait_time_steps_mean = np.mean(wait_time_steps)
            wait_time_episodes.append(wait_time_steps_mean)

            speed_steps_mean = np.mean(speed_steps)
            speed_episodes.append(speed_steps_mean)

            pedestrian_wait_mean = np.mean(pedestrian_wait_steps)
            pedestrian_wait_episodes.append(pedestrian_wait_mean)

            queue_length_steps_std = np.std(queue_length_steps)
            queue_length_episodes_std.append(queue_length_steps_std)

            wait_time_steps_std = np.std(wait_time_steps)
            wait_time_episodes_std.append(wait_time_steps_std)

            speed_steps_std = np.std(speed_steps)
            speed_episodes_std.append(speed_steps_std)

            pedestrian_wait_std = np.std(pedestrian_wait_steps)
            pedestrian_wait_episodes_std.append(pedestrian_wait_std)

            if args.verbose:

                t = np.arange(0., len(queue_length_steps))

                CustomPlot.save_plot(f'{QUEUE_PATH}Queue-Length-{str(curr_episode)}.png',
                                     'Simulation time (sec)', 'Queue Length', f'Episode :{str(curr_episode)}',
                                     t, queue_length_steps, algo)

                CustomPlot.save_plot(f'{WAIT_PATH}Wait-Time-{str(curr_episode)}.png',
                                     'Simulation time (sec)', 'Wait Time', f'Episode :{str(curr_episode)}',
                                     t, wait_time_steps, algo)

                CustomPlot.save_plot(f'{SPEED_PATH}Speed-{str(curr_episode)}.png',
                                     'Simulation time (sec)', 'Speed', f'Episode :{str(curr_episode)}',
                                     t, speed_steps, algo)

                CustomPlot.save_plot(f'{PEDESTRIAN_PATH}Pedestrian-{str(curr_episode)}.png',
                                'Simulation time (sec)', 'Pedestrian Wait Time', f'Episode:{str(curr_episode)}',
                                t, pedestrian_wait_steps, algo)

            curr_episode += 1

        if args.ci:
            queue_length_episodes_std = np.array(queue_length_episodes_std)
            queue_length_episodes = np.array(queue_length_episodes)
            std_err_que = queue_length_episodes_std / np.sqrt(timesteps)
            std_err_que *= 1.96
            CustomPlot.save_ci_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_CI_{args.algo}.png',
                                    'Episode', 'Queue Length', 'Queue Length over Episodes',
                                    t2, queue_length_episodes, queue_length_episodes-std_err_que,
                                    queue_length_episodes+std_err_que, algo)
            std_err_que_algos.append(std_err_que)
        else:
            CustomPlot.save_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_{args.algo}.png',
                                 'Episode', 'Queue Length', 'Queue Length over Episodes',
                                 t2, queue_length_episodes, algo)

        queue_length_episodes_algos.append(queue_length_episodes)

        if args.ci:
            wait_time_episodes_std = np.array(wait_time_episodes_std)
            wait_time_episodes = np.array(wait_time_episodes)
            std_err_wait = wait_time_episodes_std / np.sqrt(timesteps)
            std_err_wait *= 1.96
            CustomPlot.save_ci_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_CI_{args.algo}.png',
                                    'Episode', 'Wait Time', 'Wait Time over Episodes',
                                    t2, wait_time_episodes, wait_time_episodes-std_err_wait, wait_time_episodes
                                    + std_err_wait, algo)
            std_err_wait_algos.append(std_err_wait)
        else:
            CustomPlot.save_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_{args.algo}.png',
                                 'Episode', 'Wait Time', 'Wait Time over Episodes',
                                 t2, wait_time_episodes, algo)

        wait_time_episodes_algos.append(wait_time_episodes)

        CustomPlot.save_plot(f'{SPEED_PATH}Avg-Speed-Over_Episodes_{args.algo}.png',
                             'Episode', 'Avg. Speed', 'Avg. Speed over Episodes',
                             t2, speed_episodes, algo)

        speed_episodes_algos.append(speed_episodes)

        if args.ci:
            pedestrian_wait_episodes_std = np.array(pedestrian_wait_episodes_std)
            pedestrian_wait_episodes = np.array(pedestrian_wait_episodes)
            std_err_pedestrian_wait = pedestrian_wait_episodes_std / np.sqrt(timesteps)
            std_err_pedestrian_wait *= 1.96
            CustomPlot.save_ci_plot(f'{PEDESTRIAN_PATH}Pedestrian_Wait_time-Over_Episodes_CI_{args.algo}.png',
                                    'Episode', 'Pedestrian Wait Time', 'Pedestrian Wait Time over Episodes',
                                    t2, pedestrian_wait_episodes, pedestrian_wait_episodes-std_err_pedestrian_wait, pedestrian_wait_episodes
                                    + std_err_pedestrian_wait, algo)
            std_err_pedestrian_algos.append(std_err_pedestrian_wait)
        else:
            CustomPlot.save_plot(f'{PEDESTRIAN_PATH}Pedestrian_Wait_time-Over_Episodes_{args.algo}.png',
                                 'Episode', 'Pedestrian Wait Time', 'Pedestrian Wait Time over Episodes',
                                 t2, pedestrian_wait_episodes, algo)

        pedestrian_episodes_algos.append(pedestrian_wait_episodes)

    if args.ci:
        CustomPlot.save_combined_ci_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_CI_Combined.png',
                                         'Episode', 'Queue Length', 'Queue Length over Episodes',
                                         t2, queue_length_episodes_algos, algo_list, std_err_que_algos)
    else:
        CustomPlot.save_combined_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_Combined.png',
                                      'Episode', 'Queue Length', 'Queue Length over Episodes',
                                      t2, queue_length_episodes_algos, algo_list)

    if args.ci:
        CustomPlot.save_combined_ci_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_CI_Combined.png',
                                         'Episode', 'Wait Time', 'Wait Time over Episodes',
                                         t2, wait_time_episodes_algos, algo_list, std_err_wait_algos)
    else:
        CustomPlot.save_combined_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_Combined.png',
                                      'Episode', 'Wait Time', 'Wait Time over Episodes',
                                      t2, wait_time_episodes_algos, algo_list)

    CustomPlot.save_combined_plot(f'{SPEED_PATH}Avg-Speed-Over_Episodes_Combined.png',
                                  'Episode', 'Avg. Speed', 'Avg. Speed over Episodes',
                                  t2, speed_episodes_algos, algo_list)

    if args.ci:
        CustomPlot.save_combined_ci_plot(f'{PEDESTRIAN_PATH}Pedestrian-Wait-Time-Over_Episodes_CI_Combined.png',
                                         'Episode', 'Pedestrian Wait Time', 'Pedestrian Wait Time Over Episodes',
                                         t2, pedestrian_episodes_algos, algo_list, std_err_pedestrian_algos)
    else:
        CustomPlot.save_combined_plot(f'{PEDESTRIAN_PATH}Pedestrian-Wait-Time-Over_Episodes_Combined.png',
                                      'Episode', 'Pedestrian Wait Time', 'Pedestrian Wait Time Over Episodes',
                                      t2, pedestrian_episodes_algos, algo_list)


def visualize(args):
    agent, env_config = parse_args_and_init_agent_env(args)

    env_config['env_config']['gui'] = True
    env_config['env_config']['sumo_connector'] = 'traci'

    if args.vc_mode:
        env_module = importlib.import_module('envs.' + args.env + '_wvc')
    else:
        env_module = importlib.import_module('envs.' + args.env)

    env = env_module.TrafficSimulation(env_config['env_config'])

    obs = env.reset()
    done = False
    print_bool = args.print_info
    # time.sleep(5)
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)

        if print_bool:
            print(f"obs: {obs}\nreward: {reward}\ndone: {done}\ninfo: {info}")
            print()


if __name__ == '__main__':
    args = parse_args()
    if args.modus == 'evaluate':
        evaluate(args)
    else:
        visualize(args)
