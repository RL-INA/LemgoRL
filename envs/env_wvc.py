"""
This script maps the Gym custom environment to the Traffic Simulation
environment of SUMO, with the Lisa+ Virtual Controller in loop.
TrafficSimulatorWvcBase is the base implementation of simulation environment.
Changes are in init, reset and step method.
@author: Vishal Rangras (Fraunhofer IOSB-INA in Lemgo)
@email: vishal.rangras@iosb-ina.fraunhofer.de
"""
import logging
import getpass
from sumolib import checkBinary

import gym
from gym.spaces import Discrete, Box
import numpy as np
import traci.exceptions
from lisa_interface.middleware import LisaInterfaceManager
import yaml
import psutil


class Node:
    """
    A trafficlight/intersection is described as a Node
    """
    def __init__(self, name):
        self.lanes_in = []
        self.lanearea_detectors_in = []
        self.name = name
        self.cur_wave = []
        self.cur_wait = []
        self.cur_cum_wait = []
        self.cur_queue = []
        self.cur_avg_speed = []
        # self.phases = Phases()
        self.cur_phase_id = -1
        self.len_lanes_in = None
        self.ns_pedestrian_wait_time = 0
        self.ew_pedestrian_wait_time = 0


class TrafficSimulatorWvcBase(gym.Env):

    def __init__(self, sumo_dict, mdp_dict, sumo_env_config_dir, gui, sumo_connector, env_config,
                 worker_index, print_metrics):
        global traci
        self.env_config = env_config
        if gui or getpass.getuser() == 'arthur' or sumo_connector == 'traci':
            import traci
        else:
            import libsumo as traci

        self.print_metrics = print_metrics

        self.sumo_init_first_time = True
        self.sumo_dict = sumo_dict
        self.mdp_dict = mdp_dict

        # Give every worker a different seed_offset
        self.sumo_dict['seed'] += worker_index * 1000

        if self.env_config['env'] == 'lemgo' or self.env_config['env'] == 'owl322':
            self.controlled_nodes = sorted(env_config['controlled_nodes'])

        action_space = 0
        obs_space = 0

        if self.env_config['env'] == 'lemgo':
            action_space = 4
            if len(self.controlled_nodes) == 2:
                obs_space = 116
            elif len(self.controlled_nodes) == 1:
                obs_space = 58
        else:
            action_space = 2
            obs_space = 26
        self.action_space = Discrete(action_space)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_space,),
            dtype=np.float16
        )

        self.episodes = 0
        self.cur_timesteps = 0
        self.cur_interactionsteps = 0
        # self.cum_wait_all_veh = 0

        # Parameters Psychological emphasis of waiting time
        self.T0 = 450  # beginning of punishment for waiting in s
        self.T1 = 90  # time when P (punishment value) should be reached for waiting in s
        self.P = 5  # punishment value (reward) for waiting until T1
        self.C = self.P / (self.T1 - self.T0)**2  # multiplication factor for psychological emphasis function

        self._init_sumo(sumo_env_config_dir, gui=gui)
        self._init_nodes()
        self.owl322_lanes_dict = {
            'EntruperWeg.N.3_1': 'EntruperWeg.N.2_1',
            'EntruperWeg.N.3_2': 'EntruperWeg.N.2_2',
            'EntruperWeg.N.3_3': 'EntruperWeg.N.2_3',
            'EntruperWeg.S.8_1': 'EntruperWeg.S.7_1',
            'EntruperWeg.S.8_2': 'EntruperWeg.S.7_2',
            'EntruperWeg.S.8_3': 'EntruperWeg.S.7_3'
        }

        lisa_params = env_config['lisa_params']
        lisa_cfg = yaml.safe_load(open(lisa_params['lisa_cfg_file'], 'r'))
        self.lisa_ctx = LisaInterfaceManager(host=lisa_params['host'], port=lisa_params['port'],
                                             server_path=lisa_params['server_path'], data_dir=lisa_params['data_dir'],
                                             controlled_nodes=self.controlled_nodes, lisa_cfg=lisa_cfg)

    def _init_nodes(self):
        """
        Initialises all intersections/tls as nodes object.
        """
        nodes = {}
        for node_name in traci.trafficlight.getIDList():
            if self.env_config['env'] == 'lemgo':
                is_controlled_node = False
                for controlled_node in self.controlled_nodes:
                    if node_name == controlled_node:
                        is_controlled_node = True
                if not is_controlled_node:
                    continue

            nodes[node_name] = Node(node_name,)
            nodes[node_name].lanes_in = sorted(tuple(set(traci.trafficlight.getControlledLanes(node_name))))
            nodes[node_name].len_lanes_in = len(nodes[node_name].lanes_in)
            # LaneAreaDetectors are defined in sumo.add.xml.
            # Detector names must equal lane names.
            nodes[node_name].lanearea_detectors_in = nodes[node_name].lanes_in
            # nodes[node_name].phases =

        self.nodes = nodes
        self.node_names = sorted(list(nodes.keys()))

        logging.info(f'Initialized {len(self.node_names)} nodes.')

    def _init_sumo(self, config_dir=None, reset=False, gui=False):
        # start simulation and run the python run-function, which controls all interactions python <-> sumo
        if reset:
            traci.load(self.sumocmd[1:] + ["--seed", str(self.sumo_dict['seed'])])
            # sumoBinary darf hier nicht mehr auftauchen
        else:
            # Close sumo instances in case some ore running
            try:
                traci.close()
            except Exception as e:
                logging.error("An exception occurred. {0}".format(e))
            # Checking if --nogui option was chosen
            if gui:
                sumo_binary = checkBinary('sumo-gui')
                gui_add = ["--start", "false"]
            else:
                sumo_binary = checkBinary('sumo')
                gui_add = []
            sumocfg_path = config_dir + r"/sumo.sumocfg"
            # sumoBinary = checkBinary('sumo')
            logging.info(f'Use Sumo Configuration file {sumocfg_path}')
            self.sumocmd = [sumo_binary,
                            "-c", sumocfg_path,
                            "--tripinfo-output", "tripinfo.xml",
                            "--time-to-teleport", str(self.sumo_dict['time_to_teleport']),
                            "--step-length", str(self.sumo_dict['step_length']),
                            "--waiting-time-memory", str(self.sumo_dict['waiting_time_memory']),
                            "--no-warnings", str(self.sumo_dict['no_warnings']),
                            "--duration-log.disable", str(self.sumo_dict['duration_log_dissable']),
                            "--no-step-log", str(self.sumo_dict['no_step_log']),
                            "--default.carfollowmodel", str(self.sumo_dict['default_carfollowmodel']),
                            ] + gui_add
            traci.start(self.sumocmd + ["--seed", str(self.sumo_dict['seed'])])

        self.cur_timesteps = 0
        self.cur_interactionsteps = 0
        # In order to bring in some randomness and make every episode different, seed will be changed
        if self.sumo_dict['use_seed']:
            self.sumo_dict['seed'] += 1

    def _measure_traffic_metrics(self, phases_string_dict):
        """
        Measure specific traffic metries:
        queue length, wait time and wave for every lane.
        """
        for node_name in self.node_names:
            node = self.nodes[node_name]
            if phases_string_dict is not None:
                phases_string = phases_string_dict[node_name]
                if phases_string is not None and phases_string is not "":
                    phases_vector = [int(ph.strip()) for ph in phases_string.split("/")]
                    if phases_vector.__contains__(1):
                        node.cur_phase_id = phases_vector.index(1) + 1
                    elif phases_vector.__contains__(2):
                        node.cur_phase_id = phases_vector.index(2) + 1
                    else:
                        node.cur_phase_id = 1
                else:
                    node.cur_phase_id = 1
            else:
                node.cur_phase_id = 1

            node.cur_wave = []
            node.cur_wait = []
            node.cur_cum_wait = []
            node.cur_queue = []
            node.cur_avg_speed = []
            lanearea_detectors_in = []
            for lane_det_id in node.lanearea_detectors_in:
                try:

                    # queue length
                    queue = traci.lanearea.getLastStepHaltingNumber(lane_det_id)
                    if lane_det_id in self.owl322_lanes_dict:
                        queue += traci.lanearea.getLastStepHaltingNumber(self.owl322_lanes_dict[lane_det_id])
                    node.cur_queue.append(queue)

                    # wave
                    wave = traci.lanearea.getLastStepVehicleNumber(lane_det_id)
                    if lane_det_id in self.owl322_lanes_dict:
                        wave += traci.lanearea.getLastStepVehicleNumber(self.owl322_lanes_dict[lane_det_id])
                    node.cur_wave.append(wave)

                    # wait
                    max_pos = 0
                    wait = 0
                    cur_cars = traci.lanearea.getLastStepVehicleIDs(lane_det_id)
                    for vid in cur_cars:
                        car_pos = traci.vehicle.getLanePosition(vid)
                        if self.print_metrics:
                            print(f"vid {vid}, car_pos: {car_pos}")
                        if car_pos > max_pos:
                            max_pos = car_pos
                            wait = traci.vehicle.getWaitingTime(vid)
                    if lane_det_id in self.owl322_lanes_dict:
                        cur_cars_more = traci.lanearea.getLastStepVehicleIDs(self.owl322_lanes_dict[lane_det_id])
                        for veh in cur_cars_more:
                            car_pos = traci.vehicle.getLanePosition(veh)
                            if self.print_metrics:
                                print(f"vid {veh}, car_pos: {car_pos}")
                            if car_pos > max_pos:
                                max_pos = car_pos
                                wait = traci.vehicle.getWaitingTime(veh)
                    node.cur_wait.append(wait)

                    # cum wait - maximal accumulated waiting time of a car in a lane
                    max_cum_wait = 0
                    cur_cars = traci.lanearea.getLastStepVehicleIDs(lane_det_id)
                    for vid in cur_cars:
                        cur_cum_wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                        if cur_cum_wait > max_cum_wait:
                            max_cum_wait = cur_cum_wait
                    if lane_det_id in self.owl322_lanes_dict:
                        cur_cars_more = traci.lanearea.getLastStepVehicleIDs(self.owl322_lanes_dict[lane_det_id])
                        for veh in cur_cars_more:
                            cur_cum_wait = traci.vehicle.getAccumulatedWaitingTime(veh)
                            if cur_cum_wait > max_cum_wait:
                                max_cum_wait = cur_cum_wait
                    node.cur_cum_wait.append(max_cum_wait)

                    # avg_speed
                    avg_speed = traci.lanearea.getLastStepMeanSpeed(lane_det_id)
                    if lane_det_id in self.owl322_lanes_dict:
                        avg_speed += traci.lanearea.getLastStepMeanSpeed(self.owl322_lanes_dict[lane_det_id])
                        avg_speed = avg_speed/2
                    node.cur_avg_speed.append(avg_speed)

                    # Edge processing
                    edge_ns_list = ['Richard-Wagner-Strasse.E.1', 'Richard-Wagner-Strasse.W.6',
                                    'Gosebrede.S.1', 'Gosebrede.N.8']

                    edge_ew_list = ['EntruperWeg.N.4', 'EntruperWeg.S.8',
                                    'EntruperWeg.S.9', 'EntruperWeg.N.3']

                    if node.cur_phase_id == 6:
                        node.ns_pedestrian_wait_time = 0
                    elif node.cur_phase_id == 2:
                        node.ew_pedestrian_wait_time = 0

                    for edge in edge_ns_list:
                        person_ns_id_list = traci.edge.getLastStepPersonIDs(edge)
                        for person_ns in person_ns_id_list:
                            node.ns_pedestrian_wait_time += traci.person.getWaitingTime(person_ns)
                        if node.ns_pedestrian_wait_time > 0:
                            break

                    for edge in edge_ew_list:
                        person_ew_id_list = traci.edge.getLastStepPersonIDs(edge)
                        for person_ew in person_ew_id_list:
                            node.ew_pedestrian_wait_time += traci.person.getWaitingTime(person_ew)
                        if node.ew_pedestrian_wait_time > 0:
                            break

                    # creating new list of lanearea detectors which doesn't throw exception
                    lanearea_detectors_in.append(lane_det_id)

                    if self.print_metrics:
                        print(f'lane_det_id {lane_det_id}')
                        print(f"queue {queue}")
                        print(f"wave {wave}")
                        print(f"wait {wait}")
                        print(f"max_cum_wait {max_cum_wait}")
                        print(f"avg_speed {avg_speed}")
                        print()

                except traci.exceptions.TraCIException as e:
                    logging.error("An exception occurred. {0}".format(e))

            node.lanearea_detectors_in = lanearea_detectors_in
            node.lanes_in = node.lanearea_detectors_in
            node.len_lanes_in = len(node.lanes_in)

    def reset(self):
        if not self.sumo_init_first_time:
            self.episodes += 1
            self._init_sumo(reset=True)
        self.sumo_init_first_time = False
        # self.cum_wait_all_veh = 0

        init_phase_dict = {x: 0 for x in self.controlled_nodes}
        self.lisa_ctx.initialize_lisa_context(init_phase_dict)

        self._measure_traffic_metrics(None)
        state, _, _, _, _ = self._calc_state()
        return state

    def step(self, action):
        interval_rest = 1

        node_desired_ph = {}

        for node_name in self.node_names:
            if type(action) == dict:
                desired_action = action[node_name]
            else:
                desired_action = action
            node_desired_ph[node_name] = desired_action

        phases_string_dict = {}

        for node_name, desired_phase in node_desired_ph.items():
            sumo_signal_str, phases_string, output_string, ap_string = \
                self.lisa_ctx.get_sgr_states(node_name, desired_phase, None, self.cur_timesteps)
            traci.trafficlight.setRedYellowGreenState(node_name, sumo_signal_str)
            phases_string_dict[node_name] = phases_string

        self._simulate(interval_rest)

        if (traci.simulation.getMinExpectedNumber() > 0) \
                and (self.cur_timesteps < self.mdp_dict['max_simulation_time_sec']):
            done = False
        else:
            done = True

        self._measure_traffic_metrics(phases_string_dict)
        state, cum_waits, waves, queue_lengths, phase_id = self._calc_state()
        reward = self._calc_reward()
        self.cur_interactionsteps += 1
        return state, reward, done, dict(waves=waves, cum_waits=cum_waits, queues=queue_lengths, phase_id=phase_id)

    def close(self):
        traci.close()

        # Close Java process started for Lisa+ Virtual Controller
        for proc in psutil.process_iter():
            # Check if process name contains the given name string.
            if "javaw.exe" in proc.name().lower() or "java.exe" in proc.name().lower():
                proc.kill()

    def _simulate(self, num_steps):
        for _ in range(num_steps):
            traci.simulationStep()
            self.cur_timesteps += 1

    def _calc_state(self):
        """
        Calculate for every node the state.

        :return: reward (np.array)
        """

        queue_lengths_arr = np.array([])
        waves_arr = np.array([])
        cum_waits_arr = np.array([])
        avg_speed_arr = np.array([])
        phaseid_one_hot_encoded_arr = np.array([[]])
        pedestrian_demand_arr = np.array([[]])

        count = 0

        for node_name in self.node_names:
            # clipping and normalising
            queue_lengths = (np.array(self.nodes[node_name].cur_queue) / self.mdp_dict['queue_norm']).clip(min=0, max=2)
            queue_lengths_arr = np.concatenate([queue_lengths_arr, queue_lengths])

            waves = (np.array(self.nodes[node_name].cur_wave) / self.mdp_dict['wave_norm']).clip(min=0, max=2)
            waves_arr = np.concatenate([waves_arr, waves])

            # !!!! DEFINTION OF WAIT EITHER ACCUMULATED OR NOT
            # NOT ACCUMULATED VERSION
            # waits = (np.array(self.nodes[node_name].cur_wait) / self.mdp_dict['wait_norm']).clip(min=0, max=2)
            # ACCUMULATED VERSION
            cum_waits = (np.array(self.nodes[node_name].cur_cum_wait) / self.mdp_dict['wait_norm']).clip(min=0, max=2)
            cum_waits_arr = np.concatenate([cum_waits_arr, cum_waits])

            avg_speed = np.nan_to_num(x=(np.array(self.nodes[node_name].cur_avg_speed)
                                         / self.mdp_dict['avg_speed_norm']).clip(min=0, max=2), nan=0.0)
            avg_speed_arr = np.concatenate([avg_speed_arr, avg_speed])

            if node_name == 'Bi308' or node_name == 'OWL322':
                phaseid_one_hot_encoded = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                phaseid_one_hot_encoded[self.nodes[node_name].cur_phase_id] = 1

                if count == 0:
                    phaseid_one_hot_encoded_arr = phaseid_one_hot_encoded
                    pedestrian_demand_arr = [self.nodes[node_name].ns_pedestrian_wait_time,
                                             self.nodes[node_name].ew_pedestrian_wait_time]
                else:
                    phaseid_one_hot_encoded_arr = np.concatenate(
                        ([phaseid_one_hot_encoded_arr, phaseid_one_hot_encoded]))
                    pedestrian_demand_arr = np.concatenate(
                        (pedestrian_demand_arr, [self.nodes[node_name].ns_pedestrian_wait_time,
                                             self.nodes[node_name].ew_pedestrian_wait_time]))

            else:
                phaseid_one_hot_encoded = np.array([0, 0])
                if self.nodes[node_name].cur_phase_id == 0:
                    phaseid_one_hot_encoded[0] = 1
                else:
                    phaseid_one_hot_encoded[1] = 1

                if count == 0:
                    phaseid_one_hot_encoded_arr = phaseid_one_hot_encoded
                else:
                    phaseid_one_hot_encoded_arr = np.concatenate(
                        ([phaseid_one_hot_encoded_arr, phaseid_one_hot_encoded]))

            count += 1

        return np.concatenate([cum_waits_arr, waves_arr, queue_lengths_arr,
                               avg_speed_arr, phaseid_one_hot_encoded_arr, pedestrian_demand_arr]), waves_arr, \
            cum_waits_arr, queue_lengths_arr, phaseid_one_hot_encoded_arr

    def _calc_cum_wait_all_veh(self):
        acc_wait_all_veh = 0
        for node_name in self.node_names:
            for lane_det_id in self.nodes[node_name].lanearea_detectors_in:
                for vid in traci.lanearea.getLastStepVehicleIDs(lane_det_id):
                    acc_wait_all_veh += traci.vehicle.getAccumulatedWaitingTime(vid)
        acc_wait_all_veh = acc_wait_all_veh / self.mdp_dict['wait_norm']
        acc_wait_all_veh = acc_wait_all_veh / self.mdp_dict['all_veh_wait_norm']
        r_thrs = 4.0
        if acc_wait_all_veh > r_thrs:
            acc_wait_all_veh = r_thrs
        elif acc_wait_all_veh < -r_thrs:
            acc_wait_all_veh = -r_thrs
        return acc_wait_all_veh

    def _calc_reward(self):
        """
        Calculate for every node the reward.
        IMPORTANT! At the moment, only the reward for one_intersection environment is given.

        :return: reward (float, scalar)
        """
        reward = 0
        # reward_mapping_function = True
        for node_name in self.node_names:
            queue = np.array(self.nodes[node_name].cur_queue).sum()\
                    / self.nodes[node_name].len_lanes_in / self.mdp_dict['queue_norm']
            psych_cum_wait = np.array(self.nodes[node_name].cur_cum_wait) - self.T0
            psych_cum_wait[psych_cum_wait < 0] = 0
            psych_cum_wait = psych_cum_wait**2 * self.C
            cum_wait = psych_cum_wait.sum() / self.nodes[node_name].len_lanes_in
            reward = -1 * (queue + self.mdp_dict['wait_reward_coef'] * cum_wait)
            reward = reward.clip(min=-3, max=3)
        return reward

    def getQueueLengthsNodeMean(self):
        queue_length_nodes = []
        for node_name in self.node_names:
            queue_lengths = self.nodes[node_name].cur_queue
            queue_lengths_mean = np.mean(queue_lengths)
            queue_length_nodes.append(queue_lengths_mean)
        queue_length_nodes_mean = np.mean(queue_length_nodes)
        return queue_length_nodes_mean

    def getWaitTimeNodeMean(self):
        wait_time_nodes = []
        for node_name in self.node_names:
            wait_time = self.nodes[node_name].cur_wait
            wait_time_mean = np.mean(wait_time)
            wait_time_nodes.append(wait_time_mean)
        wait_time_nodes_mean = np.mean(wait_time_nodes)
        return wait_time_nodes_mean

    def getSpeedNodeMean(self):
        speed_nodes = []
        for node_name in self.node_names:
            speed = self.nodes[node_name].cur_avg_speed
            speed_mean = np.mean(speed)
            speed_nodes.append(speed_mean)
        speed_nodes_mean = np.mean(speed_nodes)
        return speed_nodes_mean