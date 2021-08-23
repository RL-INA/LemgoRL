"""
Copyright 2021 Arthur Mueller and Vishal Rangras
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
from custom_metrics_and_callbacks import MyCallbacks
from utils import CustomPlot
from greedy import LongestWaveFirstPolicyLemgoWvc,\
    LongestQueueFirstPolicyLemgoWvc, FixedTimeWvc, RandomTimeWvc, AdaptivePolicy
import numpy as np
from pathlib import Path
import time
from ray.rllib import agents
import lisa_interface.tsc_util as tu
import os
import ray
from collections import Counter
import evaluation.evaluationutil as eu


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-main-'+timestamp+'.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()

BASE_PATH = '.'
RESULTS_PATH = './results/'
QUEUE_RESULTS_PATH = '/queue-results/'
WAIT_RESULTS_PATH = '/wait-results/'
SPEED_RESULTS_PATH = '/speed-results/'
PHASE_WAVE_RESULT_PATH = '/phase-wave-results/'
PEDESTRIAN_RESULT_PATH = '/pedestrian-wait-results/'
REWARD_RESULTS_PATH = '/reward-results/'
CUM_REWARD_RESULTS_PATH = '/cum_reward-results/'
AVG_OVR_EP_PATH = '/avg_over_ep-results/'
VC_JAR_NAME = 'OmlFgServer.jar'

WORKER_PORT_DICT = {1: 59081, 2: 59082, 3: 59083, 4: 59084, 5: 59085, 6: 59086, 7: 59087, 8: 59088,
                    9: 59089, 10: 59090, 11: 59091, 12: 59092, 13: 59093, 14: 59094, 15: 59095, 16: 59096,
                    17: 59097, 18: 59098, 19: 59099, 20: 59100, 21: 59101, 22: 59102, 23: 59103, 24: 59104,
                    25: 59105, 26: 59106, 27: 59107, 28: 59108, 29: 59109, 30: 59110, 31: 59111, 32: 59112}


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
                        default='./lisa_vissim_addon',
                        help="Location for Lisa+ Virtual Controller server path")
    parser.add_argument('--data-dir',
                        type=str,
                        required=False,
                        default='/virtual_controller/Lemgo_OWL322_2021.07.08_VISSIM/',
                        help="Data directory for Lisa+ Virtual Controller program files")
    parser.add_argument('--lisa-cfg-file',
                        type=str,
                        required=False,
                        default='./lisa_interface/lisa_conf/lisa_config.yaml',
                        help="Path to Lisa+ Virtual Controller configuration file.")
    parser.add_argument('--sumo-connector',
                        type=str,
                        required=False,
                        default='traci',
                        help="Choose sumo-connector from traci and libsumo. Default is traci.",
                        choices=['traci', 'libsumo'])
    parser.add_argument('--starting-port',
                        type=int,
                        required=False,
                        default=59081,
                        help="Starting port number for lisa+ instances")

    subparsers = parser.add_subparsers(dest='modus',
                                       help="train, evaluate or visualize")

    subp_evaluate = subparsers.add_parser('evaluate', help="evaluate agents")

    subp_evaluate.add_argument('--algo',
                               type=str, action='append',
                               required=False,
                               help="Algorithm to evaluate the agent. default=greedy. "
                                    "Choices are from 'greedy','greedy-wave', 'PPO', 'fixed', 'random', 'adaptive'")
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
                          choices=['PPO', 'greedy','greedy-wave', 'fixed', 'random', 'adaptive'],
                          help="LemgoRL currently supports greedy policy as default algorithm.")
    subp_vis.add_argument('--mdp_sumo_id', type=str, required=False, default='last',
                          help="ID to choose the parameters for MDP and sumo in "
                               "corresponding config-file. default=last")
    subp_vis.add_argument('--print_info', type=bool, required=False, default=False,
                          help="Print observation, reward, done, info from environment. default=False")
    subp_vis.add_argument('--debug_metrics', type=bool, required=False, default=False,
                          help="Print metric values inside environment. default=False")
    subp_vis.add_argument('--analysis-logs', type=bool, required=False, default=False,
                          help="Print additional logs for analysis")

    args = parser.parse_args()
    if not args.modus:
        parser.print_help()
        exit(1)
    return args


def init_agent(algo, algo_config, env):
    """
    :param algo: algorithm to be trained on or inferred from
    :param algo_config: configuration of algorithm and env_config according to ray[rllib]
    :param env: env class; constructor of this class must take a env_config with type dict
    :return: agent instance
    """
    if algo == 'PPO':
        agent = agents.ppo.ppo.PPOTrainer(env=env, config=algo_config)
    return agent


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

    env_module = importlib.import_module('envs.' + args.env + '_wvc')

    # create env_config without gui option
    env_config = {
        'env_config': {
            'sumo_dict': sumo_dict,
            'mdp_dict': mdp_dict,
            'sumo_env_config_dir': sumo_env_config_dir,
        }
    }
    env_config['env_config']['env'] = args.env
    env_config['env_config']['controlled_nodes'] = args.controlled_nodes
    env_config['env_config']['sumo_connector'] = args.sumo_connector
    if args.modus == 'evaluate' or args.modus == 'visualize':
        env_config['env_config']['skip_worker'] = True
    else:
        env_config['env_config']['skip_worker'] = False

    if args.debug_metrics:
        env_config['env_config']['print_metrics'] = True
    else:
        env_config['env_config']['print_metrics'] = False

    if args.modus == 'visualize' and args.analysis_logs:
        env_config['env_config']['analysis_logs'] = True
    else:
        env_config['env_config']['analysis_logs'] = False

    env_config['env_config']['algo'] = args.algo

    env_config['env_config']['lisa_params'] = {
        'host': args.host,
        'port': args.port,
        'server_path': args.server_path,
        'data_dir': os.getcwd()+args.data_dir,
        'lisa_cfg_file': args.lisa_cfg_file
    }

    if args.algo == 'greedy' or args.algo == 'greedy-wave':
        state_space_config = {
            'queue_start_ind': 24,
            'wave_start_ind': 12
        }
        env_config['env_config']['state_space_config'] = state_space_config
        if args.algo == 'greedy':
            agent = LongestQueueFirstPolicyLemgoWvc(env_config['env_config'])
        else:
            agent = LongestWaveFirstPolicyLemgoWvc(env_config['env_config'])
        env_config['env_config']['gui'] = args.gui
        algo_config = {**env_config}
        if args.modus == 'evaluate' or args.modus == 'visualize':
            algo_config['num_workers'] = 1
        tu.start_oml_fg_server(algo_config, args.server_path, args.starting_port)
    elif args.algo == 'fixed' or args.algo == 'random' or args.algo == 'adaptive':
        if args.algo == 'fixed':
            agent = FixedTimeWvc(env_config['env_config'])
        elif args.algo == 'random':
            agent = RandomTimeWvc(env_config['env_config'])
        else:
            agent = AdaptivePolicy(env_config['env_config'])
        env_config['env_config']['gui'] = args.gui
        algo_config = {**env_config}
        if args.modus == 'evaluate' or args.modus == 'visualize':
            algo_config['num_workers'] = 1
        tu.start_oml_fg_server(algo_config, args.server_path, args.starting_port)
    else:
        algo_param_path = args.base_dir + r'/config/algo_parameters_' + args.algo + '.xlsx'
        algo_dict = parse_algo_parameters(excel_config_path=algo_param_path, param_group='algo',
                                          desired_id=args.algo_id)
        model_dict = parse_algo_parameters(excel_config_path=algo_param_path,
                                           param_group='model', desired_id=args.algo_id)
        model_dict = {'model': model_dict}
        algo_dict = {**algo_dict, **model_dict}

        if args.modus == 'evaluate':
            algo_dict['num_workers'] = 1

        if args.modus == 'visualize':
            algo_dict['num_workers'] = 1
            env_config['env_config']['gui'] = False
        else:
            env_config['env_config']['gui'] = args.gui

        tu.start_oml_fg_server(algo_dict, args.server_path, args.starting_port)
        time.sleep(10)
        env_config['env_config']['starting_port'] = args.starting_port

        # init agent
        log_to_driver = True
        if args.modus == 'visualize':
            log_to_driver = True

        ray.init(log_to_driver=log_to_driver)

        algo_config = {**algo_dict, **env_config, **{'callbacks': MyCallbacks}, 'log_level': 'INFO'}
        agent = init_agent(args.algo, algo_config, env_module.TrafficSimulation)

        if args.modus == 'evaluate' or args.modus == 'visualize':
            logger.info(f'Using checkpoint {args.checkpoint_nr} to restore agent from.')
            agent.restore(f'./agents_runs/{args.env}/{args.algo}/checkpoint_{args.checkpoint_nr}'
                          f'/checkpoint-{args.checkpoint_nr}')

    return agent, env_config, algo_config


def evaluate(args):
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    QUEUE_PATH = RESULTS_PATH + timestamp + QUEUE_RESULTS_PATH
    WAIT_PATH = RESULTS_PATH + timestamp + WAIT_RESULTS_PATH
    SPEED_PATH = RESULTS_PATH + timestamp + SPEED_RESULTS_PATH
    PHASE_WAVE_PATH = RESULTS_PATH + timestamp + PHASE_WAVE_RESULT_PATH
    PEDESTRIAN_PATH = RESULTS_PATH + timestamp + PEDESTRIAN_RESULT_PATH
    REWARD_PATH = RESULTS_PATH + timestamp + REWARD_RESULTS_PATH
    CUM_REWARD_PATH = RESULTS_PATH + timestamp + CUM_REWARD_RESULTS_PATH
    AOE_PATH = RESULTS_PATH + timestamp + AVG_OVR_EP_PATH
    Path(QUEUE_PATH).mkdir(parents=True, exist_ok=True)
    Path(WAIT_PATH).mkdir(parents=True, exist_ok=True)
    Path(SPEED_PATH).mkdir(parents=True, exist_ok=True)
    Path(PHASE_WAVE_PATH).mkdir(parents=True, exist_ok=True)
    Path(PEDESTRIAN_PATH).mkdir(parents=True, exist_ok=True)
    Path(REWARD_PATH).mkdir(parents=True, exist_ok=True)
    Path(CUM_REWARD_PATH).mkdir(parents=True, exist_ok=True)
    Path(AOE_PATH).mkdir(parents=True, exist_ok=True)

    algo_list = args.algo
    checkpoint_list = args.checkpoint_nr
    trial_info_list = [None] * len(algo_list)

    queue_length_episodes_algos = []
    wait_time_episodes_algos = []
    speed_episodes_algos = []
    pedestrian_episodes_algos = []
    std_err_que_algos = []
    std_err_wait_algos = []
    std_err_avg_speed_algos = []
    std_err_pedestrian_algos = []
    actual_phase_durations_algos_dict = {}
    desired_phase_durations_algos_dict = {}

    queue_length_episodes_list_algos = []
    wait_time_episodes_list_algos = []
    speed_episodes_list_algos = []
    pedestrian_episodes_list_algos = []
    reward_lists_algos = []
    cum_reward_lists_algos = []
    std_err_que_list_algos = []
    std_err_wait_list_algos = []
    std_err_avg_speed_algos = []
    std_err_pedestrian_list_algos = []
    std_err_reward_lists_algos = []
    std_err_cum_reward_lists_algos = []

    queue_algo_summary_list = []
    wait_algo_summary_list = []
    ped_wait_algo_summary_list = []
    speed_algo_summary_list = []
    reward_algo_summary_list = []
    cum_reward_algo_summary_list = []

    ma_queue_algo_summary_list = []
    ma_wait_algo_summary_list = []
    ma_ped_wait_algo_summary_list = []
    ma_speed_algo_summary_list = []
    ma_reward_algo_summary_list = []
    ma_cum_reward_algo_summary_list = []

    timesteps = 4200

    max_episode = args.episodes
    t2 = np.arange(1., max_episode + 1)
    tu.shutdown_oml_fg_server()

    for algo, checkpoint_nr, trial_info in zip(algo_list, checkpoint_list, trial_info_list):
        args_dict = vars(args)
        args_dict['algo'] = algo
        args_dict['checkpoint_nr'] = checkpoint_nr
        checkpoint_nr = str(checkpoint_nr)

        logger.info(f"Evaluating algo: {algo}, checkpoint_nr: {checkpoint_nr}")

        agent, env_config, algo_config = parse_args_and_init_agent_env(args)

        env_module = importlib.import_module('envs.' + args.env + '_wvc')
        env = env_module.TrafficSimulation(env_config['env_config'])

        curr_episode = 1

        queue_length_episodes = []
        wait_time_episodes = []
        speed_episodes = []
        pedestrian_wait_episodes = []
        desired_ph_episodes = []
        actual_ph_episodes = []
        desired_ph_change_episode = []
        actual_ph_change_episode = []
        ki_run_count_episode = []

        queue_length_episodes_list = []
        wait_time_episodes_list = []
        speed_episodes_list = []
        pedestrian_wait_episodes_list = []
        reward_episodes_list = []
        cum_reward_episodes_list = []

        queue_length_episodes_std = []
        wait_time_episodes_std = []
        speed_episodes_std = []
        pedestrian_wait_episodes_std = []

        CustomPlot.plot_figure()
        time.sleep(10)

        while curr_episode <= max_episode:
            logger.info(f"Evaluating episode: {curr_episode}")
            obs = env.reset()
            done = False
            step = 1
            queue_length_steps = []
            wait_time_steps = []
            speed_steps = []
            pedestrian_wait_steps = []
            reward_steps = []
            cum_reward_steps = []
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

                reward_mean = env.getRewardMean()
                reward_steps.append(reward_mean)

                cum_reward_mean = env.getCumRewardMean()
                cum_reward_steps.append(cum_reward_mean)

                step += 1

            if env_config['env_config']['env'] == 'owl322' and max_episode < 10:
                desired_ph_list, actual_ph_list = env.getPhaseCounts()
                desired_ph_episodes += desired_ph_list
                actual_ph_episodes += actual_ph_list
                actual_phase_durations_dict = env.getActualPhaseDurationDict()
                for key, value in actual_phase_durations_dict.items():
                    if key in actual_phase_durations_algos_dict:
                        actual_phase_durations_algos_dict[key].extend(value)
                    else:
                        actual_phase_durations_algos_dict[key] = value

                desired_phase_durations_dict = env.getDesiredPhaseDurationDict()
                for key, value in desired_phase_durations_dict.items():
                    if key in desired_phase_durations_algos_dict:
                        desired_phase_durations_algos_dict[key].extend(value)
                    else:
                        desired_phase_durations_algos_dict[key] = value

                desired_ph_change_count, actual_ph_change_count, ki_run_count = env.getPhaseChangeCounts()
                desired_ph_change_episode.append(desired_ph_change_count)
                actual_ph_change_episode.append(actual_ph_change_count)
                ki_run_count_episode.append(ki_run_count)

                if args.verbose:
                    PH_DETAILS_V_PATH = f'{RESULTS_PATH + timestamp}/Phase_details/{algo+checkpoint_nr}'
                    Path(PH_DETAILS_V_PATH).mkdir(parents=True, exist_ok=True)
                    ph_v_file = f'{PH_DETAILS_V_PATH}/Ph_Details-{str(curr_episode)}.txt'
                    with open(ph_v_file, "w") as text_file:
                        print(f"Desired Phase : {sorted(Counter(desired_ph_list).items())}", file=text_file)
                        print(f"Actual Phase: {sorted(Counter(actual_ph_list).items())}", file=text_file)
                        print(f"Desired Phase Change Count: {desired_ph_change_count}", file=text_file)
                        print(f"Actual Phase Change Count: {actual_ph_change_count}", file=text_file)
                        print(f"KI_Run count: {ki_run_count}", file=text_file)
                        for key, value in actual_phase_durations_dict.items():
                            print(f"Actual Phase: {key}, Min_Duration: {np.min(value)},"
                                  f" Mean_Duration: {np.mean(value)}, Max_Duration: {np.max(value)}", file=text_file)
                        for key, value in desired_phase_durations_dict.items():
                            print(f"Desired Phase: {key}, Min_Duration: {np.min(value)},"
                                  f" Mean_Duration: {np.mean(value)}, Max_Duration: {np.max(value)}", file=text_file)

            queue_length_steps_mean = np.mean(queue_length_steps)
            queue_length_episodes.append(queue_length_steps_mean)
            queue_length_episodes_list.append(queue_length_steps)

            wait_time_steps_mean = np.mean(wait_time_steps)
            wait_time_episodes.append(wait_time_steps_mean)
            wait_time_episodes_list.append(wait_time_steps)

            speed_steps_mean = np.mean(speed_steps)
            speed_episodes.append(speed_steps_mean)
            speed_episodes_list.append(speed_steps)

            pedestrian_wait_mean = np.mean(pedestrian_wait_steps)
            pedestrian_wait_episodes.append(pedestrian_wait_mean)
            pedestrian_wait_episodes_list.append(pedestrian_wait_steps)

            reward_episodes_list.append(reward_steps)
            cum_reward_episodes_list.append(cum_reward_steps)

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

                CustomPlot.save_plot(f'{QUEUE_PATH}Queue-Length_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
                                     'Simulation time (sec)', 'Queue Length (m)', f'Episode :{str(curr_episode)}',
                                     t, queue_length_steps, algo+checkpoint_nr)

                CustomPlot.save_plot(f'{WAIT_PATH}Wait-Time_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
                                     'Simulation time (sec)', 'Vehicle Wait Time (s)', f'Episode :{str(curr_episode)}',
                                     t, wait_time_steps, algo+checkpoint_nr)

                CustomPlot.save_plot(f'{SPEED_PATH}Speed_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
                                     'Simulation time (sec)', 'Speed (m/s)', f'Episode :{str(curr_episode)}',
                                     t, speed_steps, algo+checkpoint_nr)

                CustomPlot.save_plot(f'{PEDESTRIAN_PATH}Pedestrian_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
                                     'Simulation time (sec)', 'Pedestrian Wait Time (s)', f'Episode:{str(curr_episode)}',
                                     t, pedestrian_wait_steps, algo+checkpoint_nr)

                CustomPlot.save_plot(f'{REWARD_PATH}Reward_{args.algo}_{checkpoint_nr}-{str(curr_episode)}.png',
                                     'Simulation time (sec)', 'Reward', f'Episode:{str(curr_episode)}',
                                     t, reward_steps, algo + checkpoint_nr)

            curr_episode += 1

        if env_config['env_config']['env'] == 'owl322':
            PH_DETAILS_PATH = f'{RESULTS_PATH + timestamp}/Phase_details/{algo+checkpoint_nr}'
            Path(PH_DETAILS_PATH).mkdir(parents=True, exist_ok=True)
            ph_file = f'{PH_DETAILS_PATH}/Ph_Details-{algo+checkpoint_nr}.txt'
            with open(ph_file, "w") as text_file:
                print(f"Desired Phase : {sorted(Counter(desired_ph_episodes).items())}", file=text_file)
                print(f"Actual Phase: {sorted(Counter(actual_ph_episodes).items())}", file=text_file)
                print(f"Desired Phase Change Count: {np.sum(desired_ph_change_episode)}", file=text_file)
                print(f"Actual Phase Change Count: {np.sum(actual_ph_change_episode)}", file=text_file)
                print(f"KI_Run Count: {np.sum(ki_run_count_episode)}", file=text_file)
                for key, value in actual_phase_durations_algos_dict.items():
                    print(f"Total: Actual Phase: {key}, Min_Duration: {np.min(value)},"
                          f" Mean_Duration: {np.mean(value)}, Max_Duration: {np.max(value)}", file=text_file)
                for key, value in desired_phase_durations_algos_dict.items():
                    print(f"Total: Desired Phase: {key}, Min_Duration: {np.min(value)},"
                          f" Mean_Duration: {np.mean(value)}, Max_Duration: {np.max(value)}", file=text_file)

        t = np.arange(0., timesteps)

        if args.ci:
            queue_length_episodes_std = np.array(queue_length_episodes_std)
            queue_length_episodes = np.array(queue_length_episodes)
            std_err_que = queue_length_episodes_std / np.sqrt(timesteps)
            std_err_que *= 1.96
            CustomPlot.save_ci_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_CI_{args.algo}_{checkpoint_nr}.png',
                                    'Episode', 'Queue Length (m)', 'Queue Length over Episodes',
                                    t2, queue_length_episodes, queue_length_episodes-std_err_que,
                                    queue_length_episodes+std_err_que, algo+checkpoint_nr)
            std_err_que_algos.append(std_err_que)

            ma_queue_mean, ma_queue_std = eu.plot_moving_average_over_episode(queue_length_episodes_list, 'Queue Length',
                                                QUEUE_PATH, args.algo, checkpoint_nr,
                                                queue_length_episodes_list_algos, std_err_que_list_algos)

            queue_mean, queue_std = eu.compute_metric_mean_std(queue_length_episodes_list)
            queue_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Queue Length', queue_mean, queue_std))
            ma_queue_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Queue Length', ma_queue_mean, ma_queue_std))

        else:
            CustomPlot.save_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
                                 'Episode', 'Queue Length (m)', 'Queue Length over Episodes',
                                 t2, queue_length_episodes, algo+checkpoint_nr)

        queue_length_episodes_algos.append(queue_length_episodes)

        if args.ci:
            wait_time_episodes_std = np.array(wait_time_episodes_std)
            wait_time_episodes = np.array(wait_time_episodes)
            std_err_wait = wait_time_episodes_std / np.sqrt(timesteps)
            std_err_wait *= 1.96
            CustomPlot.save_ci_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_CI_{args.algo}_{checkpoint_nr}.png',
                                    'Episode', 'Vehicle Wait Time (s)', 'Wait Time over Episodes',
                                    t2, wait_time_episodes, wait_time_episodes-std_err_wait, wait_time_episodes
                                    + std_err_wait, algo+checkpoint_nr)
            std_err_wait_algos.append(std_err_wait)

            ma_wait_mean, ma_wait_std = eu.plot_moving_average_over_episode(wait_time_episodes_list, 'Vehicle Wait Time',
                                                WAIT_PATH, args.algo, checkpoint_nr,
                                                wait_time_episodes_list_algos, std_err_wait_list_algos)

            wait_mean, wait_std = eu.compute_metric_mean_std(wait_time_episodes_list)
            wait_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Vehicle Wait Time', wait_mean, wait_std))
            ma_wait_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Vehicle Wait Time', ma_wait_mean, ma_wait_std))
        else:
            CustomPlot.save_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
                                 'Episode', 'Vehicle Wait Time (s)', 'Wait Time over Episodes',
                                 t2, wait_time_episodes, algo+checkpoint_nr)

        wait_time_episodes_algos.append(wait_time_episodes)

        CustomPlot.save_plot(f'{SPEED_PATH}Avg-Speed-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
                             'Episode', 'Speed (m/s)', 'Avg. Speed over Episodes',
                             t2, speed_episodes, algo+checkpoint_nr)

        speed_episodes_algos.append(speed_episodes)

        if args.ci:
            ma_speed_mean, ma_speed_std = eu.plot_moving_average_over_episode(speed_episodes_list, 'Speed',
                                                SPEED_PATH, args.algo, checkpoint_nr,
                                                speed_episodes_list_algos, std_err_avg_speed_algos)

            speed_mean, speed_std = eu.compute_metric_mean_std(speed_episodes_list)
            speed_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Speed', speed_mean, speed_std))
            ma_speed_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Speed', ma_speed_mean, ma_speed_std))

        if args.ci:
            pedestrian_wait_episodes_std = np.array(pedestrian_wait_episodes_std)
            pedestrian_wait_episodes = np.array(pedestrian_wait_episodes)
            std_err_pedestrian_wait = pedestrian_wait_episodes_std / np.sqrt(timesteps)
            std_err_pedestrian_wait *= 1.96
            CustomPlot.save_ci_plot(f'{PEDESTRIAN_PATH}Pedestrian_Wait_time-Over_Episodes_CI_{args.algo}_{checkpoint_nr}.png',
                                    'Episode', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time over Episodes',
                                    t2, pedestrian_wait_episodes, pedestrian_wait_episodes-std_err_pedestrian_wait, pedestrian_wait_episodes
                                    + std_err_pedestrian_wait, algo+checkpoint_nr)
            std_err_pedestrian_algos.append(std_err_pedestrian_wait)

            ma_ped_wait_mean, ma_ped_wait_std = eu.plot_moving_average_over_episode(pedestrian_wait_episodes_list, 'Pedestrian Wait Time',
                                                PEDESTRIAN_PATH, args.algo, checkpoint_nr,
                                                pedestrian_episodes_list_algos, std_err_pedestrian_list_algos)

            ped_wait_mean, ped_wait_std = eu.compute_metric_mean_std(pedestrian_wait_episodes_list)
            ped_wait_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Pedestrian Wait Time', ped_wait_mean, ped_wait_std))
            ma_ped_wait_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Pedestrian Wait Time', ma_ped_wait_mean, ma_ped_wait_std))

        else:
            CustomPlot.save_plot(f'{PEDESTRIAN_PATH}Pedestrian_Wait_time-Over_Episodes_{args.algo}_{checkpoint_nr}.png',
                                 'Episode', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time over Episodes',
                                 t2, pedestrian_wait_episodes, algo+checkpoint_nr)

        pedestrian_episodes_algos.append(pedestrian_wait_episodes)

        if args.ci:
            ma_reward_mean, ma_reward_std = eu.plot_moving_average_over_episode(reward_episodes_list, 'Reward',
                                                REWARD_PATH, args.algo, checkpoint_nr,
                                                reward_lists_algos, std_err_reward_lists_algos)

            reward_mean, reward_std = eu.compute_metric_mean_std(reward_episodes_list)
            reward_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Reward', reward_mean, reward_std))
            ma_reward_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Reward', ma_reward_mean, ma_reward_std))

            ma_cum_reward_mean, ma_cum_reward_std = eu.plot_moving_average_over_episode(cum_reward_episodes_list, 'Cummulative Reward',
                                                CUM_REWARD_PATH, args.algo, checkpoint_nr,
                                                cum_reward_lists_algos, std_err_cum_reward_lists_algos)

            cum_reward_mean, cum_reward_std = eu.compute_metric_mean_std(cum_reward_episodes_list)
            cum_reward_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Reward', cum_reward_mean, cum_reward_std))
            ma_cum_reward_algo_summary_list.append(eu.AlgoSummary(args.algo, checkpoint_nr, 'Reward', ma_cum_reward_mean, ma_cum_reward_std))

        ray.shutdown()

    metric_summary_dict = {'queue': queue_algo_summary_list, 'wait_time': wait_algo_summary_list,
                           'ped_wait_time': ped_wait_algo_summary_list, 'speed': speed_algo_summary_list,
                           'reward': reward_algo_summary_list, 'cum_reward': cum_reward_algo_summary_list}

    eu.save_summary_table(RESULTS_PATH + timestamp, metric_summary_dict)

    ma_metric_summary_dict = {'queue': ma_queue_algo_summary_list, 'wait_time': ma_wait_algo_summary_list,
                              'ped_wait_time': ma_ped_wait_algo_summary_list, 'speed': ma_speed_algo_summary_list,
                              'reward': ma_reward_algo_summary_list, 'cum_reward': ma_cum_reward_algo_summary_list}

    eu.save_summary_table(RESULTS_PATH + timestamp, ma_metric_summary_dict, ma=True)

    t = np.arange(0., timesteps)

    if args.ci:
        CustomPlot.save_combined_ci_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_CI_Combined.png',
                                         'Episode', 'Queue Length (m)', 'Queue Length over Episodes',
                                         t2, queue_length_episodes_algos, algo_list, checkpoint_list, std_err_que_algos)

        CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Queue-Length-Avg_Over_Episodes_CI_Combined.png',
                                         'Simulation time (s)', 'Queue Length (m)', 'Queue Length over Timesteps',
                                         t, queue_length_episodes_list_algos, algo_list, checkpoint_list,
                                         std_err_que_list_algos)

    else:
        CustomPlot.save_combined_plot(f'{QUEUE_PATH}Queue-Length-Over_Episodes_Combined.png',
                                      'Episode', 'Queue Length (m)', 'Queue Length over Episodes',
                                      t2, queue_length_episodes_algos, algo_list, checkpoint_list)

    if args.ci:
        CustomPlot.save_combined_ci_plot(f'{WAIT_PATH}Wait_time-Over-Episodes_CI_Combined.png',
                                         'Episode', 'Vehicle Wait Time (s)', 'Wait Time over Episodes',
                                         t2, wait_time_episodes_algos, algo_list, checkpoint_list, std_err_wait_algos)

        CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Wait_time-Avg-Over-Episodes_CI_Combined.png',
                                         'Simulation time (s)', 'Vehicle Wait Time (s)', 'Wait Time over Timesteps',
                                         t, wait_time_episodes_list_algos, algo_list, checkpoint_list,
                                         std_err_wait_list_algos)
    else:
        CustomPlot.save_combined_plot(f'{WAIT_PATH}Wait_time-Over_Episodes_Combined.png',
                                      'Episode', 'Vehicle Wait Time (s)', 'Wait Time over Episodes',
                                      t2, wait_time_episodes_algos, algo_list, checkpoint_list)

    CustomPlot.save_combined_plot(f'{SPEED_PATH}Avg-Speed-Over_Episodes_Combined.png',
                                  'Episode', 'Speed (m/s)', 'Avg. Speed over Episodes',
                                  t2, speed_episodes_algos, algo_list, checkpoint_list)

    if args.ci:
        CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Speed-Avg-Over-Episodes_CI_Combined.png',
                                         'Simulation time (s)', 'Speed (m/s)', 'Avg. Speed over Timesteps',
                                         t, speed_episodes_list_algos, algo_list, checkpoint_list,
                                         std_err_avg_speed_algos)

    if args.ci:
        CustomPlot.save_combined_ci_plot(f'{PEDESTRIAN_PATH}Pedestrian-Wait-Time-Over_Episodes_CI_Combined.png',
                                         'Episode', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time Over Episodes',
                                         t2, pedestrian_episodes_algos, algo_list, checkpoint_list, std_err_pedestrian_algos)

        CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Pedestrian-Wait_time-Avg-Over-Episodes_CI_Combined.png',
                                         'Simulation time (s)', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time over Timesteps',
                                         t, pedestrian_episodes_list_algos, algo_list, checkpoint_list,
                                         std_err_pedestrian_list_algos)

        CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Pedestrian-Wait_time-Avg-Over-Episodes_CI_Combined_WO_L.png',
                                         'Simulation time (s)', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time over Timesteps',
                                         t, pedestrian_episodes_list_algos[0:-1], algo_list[0:-1], checkpoint_list[0:-1],
                                         std_err_pedestrian_list_algos[0:-1])

    else:
        CustomPlot.save_combined_plot(f'{PEDESTRIAN_PATH}Pedestrian-Wait-Time-Over_Episodes_Combined.png',
                                      'Episode', 'Pedestrian Wait Time (s)', 'Pedestrian Wait Time Over Episodes',
                                      t2, pedestrian_episodes_algos, algo_list, checkpoint_list)

    CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Reward-Avg-Over-Episodes_CI_Combined.png',
                                  'Simulation time (s)', 'Reward', 'Reward over Timesteps',
                                  t, reward_lists_algos, algo_list, checkpoint_list, std_err_reward_lists_algos)

    CustomPlot.save_combined_ci_plot(f'{AOE_PATH}Cum_Reward-Avg-Over-Episodes_CI_Combined.png',
                                     'Simulation time (s)', 'Cum Reward', 'Reward over Timesteps',
                                     t, cum_reward_lists_algos, algo_list, checkpoint_list, std_err_cum_reward_lists_algos)


def visualize(args):
    agent, env_config, algo_config = parse_args_and_init_agent_env(args)

    env_config['env_config']['gui'] = True
    env_config['env_config']['sumo_connector'] = 'traci'

    env_module = importlib.import_module('envs.' + args.env + '_wvc')

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
    try:
        args = parse_args()
        if args.modus == 'evaluate':
            evaluate(args)
        else:
            visualize(args)
    finally:
        logger.info("Shutting down the OmlFgServers...")
        tu.shutdown_oml_fg_server()
