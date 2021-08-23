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
This script maps the Gym custom environment to the Traffic Simulation
environment of SUMO, with the Lisa+ Virtual Controller in loop.
TrafficSimulatorWvcBase is the base implementation of simulation environment.
Changes are in init, reset and step method.
@author: Vishal Rangras (Fraunhofer IOSB-INA in Lemgo)
@email: vishal.rangras@iosb-ina.fraunhofer.de
"""
import logging
import time
from pathlib import Path
import getpass
from sumolib import checkBinary

import gym
from gym.spaces import Discrete, Box
import numpy as np
import traci.exceptions
from lisa_interface.middleware import LisaInterfaceManager
import yaml
import psutil

VC_JAR_NAME = 'OmlFgServer.jar'


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-envwvc-'+timestamp+'.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()

WORKER_PORT_DICT = {0: 59081, 1: 59081, 2: 59082, 3: 59083, 4: 59084, 5: 59085, 6: 59086, 7: 59087, 8: 59088,
                    9: 59089, 10: 59090, 11: 59091, 12: 59092, 13: 59093, 14: 59094, 15: 59095, 16: 59096,
                    17: 59097, 18: 59098, 19: 59099, 20: 59100, 21: 59101, 22: 59102, 23: 59103, 24: 59104,
                    25: 59105, 26: 59106, 27: 59107, 28: 59108, 29: 59109, 30: 59110, 31: 59111, 32: 59112}


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
        self.cur_phase_id = -1
        self.desired_phase_id = -1
        self.len_lanes_in = None
        self.ns_pedestrian_wait_time = 0
        self.ew_pedestrian_wait_time = 0
        self.ns_person_dict = {}
        self.ew_person_dict = {}
        self.cur_reward = 0
        self.cum_reward = 0


class TrafficSimulatorWvcBase(gym.Env):

    def __init__(self, sumo_dict, mdp_dict, sumo_env_config_dir, gui, sumo_connector, env_config,
                 worker_index, print_metrics):
        self.worker_index = worker_index
        self.lisa_port = WORKER_PORT_DICT[worker_index]
        self.analysis_logs = env_config['analysis_logs']
        self.algo = env_config['algo']
        logger.info(f'Worker Index: {self.worker_index} is assigned the port: {self.lisa_port}')
        self.lisa_pid = 0
        self.lisa_process = None
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

        if self.env_config['env'] == 'lemgo' or self.env_config['env'] == 'owl322':
            action_space = 7
            if len(self.controlled_nodes) == 2:
                obs_space = 116
            elif len(self.controlled_nodes) == 1:
                obs_space = 68
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

        if self.algo == 'adaptive':
            self.lisa_ctx = LisaInterfaceManager(host=lisa_params['host'], port=self.lisa_port,
                                                 server_path=lisa_params['server_path'],
                                                 data_dir=lisa_params['data_dir'],
                                                 controlled_nodes=self.controlled_nodes, lisa_cfg=lisa_cfg,
                                                 adaptive=True)
        else:
            self.lisa_ctx = LisaInterfaceManager(host=lisa_params['host'], port=self.lisa_port,
                                                 server_path=lisa_params['server_path'], data_dir=lisa_params['data_dir'],
                                                 controlled_nodes=self.controlled_nodes, lisa_cfg=lisa_cfg)
        self.actual_ph_list = None
        self.desired_ph_list = None
        self.prev_actual_ph = None
        self.prev_desired_ph = None
        self.actual_ph_change_count = 0
        self.desired_ph_change_count = 0
        self.ki_run_count = 0
        self.elapsed_time = 0
        self.desired_elapsed_time = 0
        self.ki_run = 0
        self.actual_phase_duration_dict = {}
        self.desired_phase_duration_dict = {}
        self.detector_type_dict = {'OWL322_e2_V010': 'e2', 'OWL322_e1_D011': 'e1', 'OWL322_e2_V210': 'e2', 'OWL322_e3_V02R': 'e3',
                                   'OWL322_e3_V020': 'e3', 'OWL322_e2_V030': 'e2', 'OWL322_e2_V230': 'e2', 'OWL322_e1_D031': 'e1',
                                   'OWL322_e1_D032': 'e1', 'OWL322_e3_V04R': 'e3', 'OWL322_e3_V040': 'e3', 'OWL322_e3_V041': 'e3'}
        self.prev_phase = None
        self.phase_count = 0
        self.active_det_dict = {}

    def _init_nodes(self):
        """
        Initialises all intersections/tls as nodes object.
        """
        nodes = {}
        for node_name in traci.trafficlight.getIDList():
            if self.env_config['env'] == 'lemgo':
                if node_name not in self.controlled_nodes:
                    continue

            nodes[node_name] = Node(node_name,)
            nodes[node_name].lanes_in = sorted(tuple(set(traci.trafficlight.getControlledLanes(node_name))))
            nodes[node_name].len_lanes_in = len(nodes[node_name].lanes_in)
            # LaneAreaDetectors are defined in sumo.add.xml.
            # Detector names must equal lane names.
            nodes[node_name].lanearea_detectors_in = nodes[node_name].lanes_in

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
            if self.algo == 'adaptive':
                sumocfg_path = config_dir + r"/sumo_adapt.sumocfg"
            else:
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

    def _measure_traffic_metrics(self, phases_string_dict, desired_phase):
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
            if desired_phase is not None:
                node.desired_phase_id = desired_phase
            else:
                node.desired_phase_id = 2

            if self.algo == 'adaptive':
                if self.prev_phase is None:
                    self.prev_phase = node.cur_phase_id
                    self.phase_count += 1
                else:
                    if self.prev_phase == node.cur_phase_id:
                        self.phase_count += 1
                    else:
                        self.phase_count = 1
                        self.prev_phase = node.cur_phase_id

            node.cur_wave = []
            node.cur_wait = []
            node.cur_cum_wait = []
            node.cur_queue = []
            node.cur_avg_speed = []
            lanearea_detectors_in = []

            # Pedestrian processing
            edge_list = [':Junction_Gosebrede/Richard-Wagner-Strasse_EntruperWeg_w0',
                         ':Junction_Gosebrede/Richard-Wagner-Strasse_EntruperWeg_w1',
                         ':Junction_Gosebrede/Richard-Wagner-Strasse_EntruperWeg_w2',
                         ':Junction_Gosebrede/Richard-Wagner-Strasse_EntruperWeg_w3']

            ns_cross_list = [':Junction_Gosebrede/Richard-Wagner-Strasse_EntruperWeg_c0',
                             ':Junction_Gosebrede/Richard-Wagner-Strasse_EntruperWeg_c2']

            ew_cross_list = [':Junction_Gosebrede/Richard-Wagner-Strasse_EntruperWeg_c1',
                             ':Junction_Gosebrede/Richard-Wagner-Strasse_EntruperWeg_c3']

            if node.cur_phase_id == 6:
                node.ns_pedestrian_wait_time = 0
                node.ns_person_dict.clear()
            elif node.cur_phase_id == 2:
                node.ew_pedestrian_wait_time = 0
                node.ew_person_dict.clear()

            for edge in edge_list:
                person_id_list = traci.edge.getLastStepPersonIDs(edge)
                for person in person_id_list:
                    person_speed = traci.person.getSpeed(person)
                    if person_speed < 0.1:
                        person_next_edge = traci.person.getNextEdge(person)
                        if person_next_edge in ns_cross_list:
                            node.ns_pedestrian_wait_time += 1
                            break
                        elif traci.person.getNextEdge(person) in ew_cross_list:
                            node.ew_pedestrian_wait_time += 1
                            break

            for lane_det_id in node.lanearea_detectors_in:
                try:

                    # queue length
                    queue = traci.lanearea.getJamLengthMeters(lane_det_id)
                    if lane_det_id in self.owl322_lanes_dict:
                        queue += traci.lanearea.getJamLengthMeters(self.owl322_lanes_dict[lane_det_id])
                    node.cur_queue.append(np.nan_to_num(queue, nan=0.0))

                    # wave
                    wave = traci.lanearea.getLastStepVehicleNumber(lane_det_id)
                    if lane_det_id in self.owl322_lanes_dict:
                        wave += traci.lanearea.getLastStepVehicleNumber(self.owl322_lanes_dict[lane_det_id])
                    node.cur_wave.append(np.nan_to_num(wave, nan=0.0))

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
                    node.cur_wait.append(np.nan_to_num(wait, nan=0.0))

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
                    node.cur_cum_wait.append(np.nan_to_num(max_cum_wait, nan=0.0))

                    # avg_speed
                    avg_speed = traci.lanearea.getLastStepMeanSpeed(lane_det_id)
                    if lane_det_id in self.owl322_lanes_dict:
                        add_speed = traci.lanearea.getLastStepMeanSpeed(self.owl322_lanes_dict[lane_det_id])
                        avg_speed += add_speed
                        avg_speed = avg_speed/2
                    node.cur_avg_speed.append(np.nan_to_num(avg_speed, nan=0.0))

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
        if self.env_config['env'] == 'owl322':
            self.actual_ph_list = []
            self.desired_ph_list = []
            self.prev_actual_ph = None
            self.prev_desired_ph = None
            self.actual_ph_change_count = 0
            self.desired_ph_change_count = 0
            self.ki_run_count = 0
            self.nodes['OWL322'].cum_reward = 0

        if not self.sumo_init_first_time:
            self.episodes += 1
            self._init_sumo(reset=True)
        self.sumo_init_first_time = False

        init_phase_dict = {x: 0 for x in self.controlled_nodes}
        self.lisa_ctx.initialize_lisa_context(init_phase_dict)

        for i in range(27):
            if self.algo == 'adaptive':
                self.lisa_ctx.get_sgr_states_det('OWL322', '', None, i)
            else:
                self.lisa_ctx.get_sgr_states('OWL322', 1, None, i)

        self._measure_traffic_metrics(None, None)
        state, _, _, _, _, _, _, _ = self._calc_state()
        return state

    def step(self, action):
        interval_rest = 1

        node_desired_ph = {}
        desire_phase_id = None
        ki_run = None
        phases_string_dict = {}

        if self.algo != 'adaptive':
            for node_name in self.node_names:
                if type(action) == dict:
                    desired_action = action[node_name]
                else:
                    desired_action = action
                if self.algo == 'greedy' or self.algo == 'greedy-wave':
                    desired_phase = desired_action + 1
                elif self.algo == 'fixed' or self.algo == 'random':
                    desired_phase = desired_action
                else:
                    # This is for RL Agent's action space (Action 0 to Action 6 become Phase 2 to 8)
                    desired_phase = desired_action + 2
                node_desired_ph[node_name] = desired_phase

        if self.algo == 'adaptive':
            for node_name in self.node_names:
                det_str = self.__get_detector_str(self.nodes[node_name].ns_pedestrian_wait_time,
                                                  self.nodes[node_name].ew_pedestrian_wait_time)
                sumo_signal_str, phases_string, output_string, ap_string = \
                    self.lisa_ctx.get_sgr_states_det(node_name, det_str, None, self.cur_timesteps + 27)
                traci.trafficlight.setRedYellowGreenState(node_name, sumo_signal_str)
                phases_string_dict[node_name] = phases_string
        else:
            for node_name, desired_phase in node_desired_ph.items():
                sumo_signal_str, phases_string, output_string, ap_string = \
                    self.lisa_ctx.get_sgr_states(node_name, desired_phase, None, self.cur_timesteps+27)
                traci.trafficlight.setRedYellowGreenState(node_name, sumo_signal_str)
                phases_string_dict[node_name] = phases_string
                desire_phase_id = desired_phase
                ki_run = ap_string.split("/")[-1]

        self._simulate(interval_rest)

        if (traci.simulation.getMinExpectedNumber() > 0) \
                and (self.cur_timesteps < self.mdp_dict['max_simulation_time_sec']):
            done = False
        else:
            done = True

        if self.env_config['env'] == 'owl322' and self.algo != 'adaptive':
            self._measure_traffic_metrics(phases_string_dict, node_desired_ph['OWL322'])
        else:
            self._measure_traffic_metrics(phases_string_dict, None)
        state, waves, cum_waits, queue_lengths, phase_id, cur_avg_speed, ped_wait_array, desired_ph_one_hot_encoded = self._calc_state()

        if self.env_config['env'] == 'owl322' and self.algo != 'adaptive':
            self.desired_ph_list.append(node_desired_ph['OWL322'])
            self.actual_ph_list.append(self.nodes['OWL322'].cur_phase_id)

            if self.prev_desired_ph is not None and self.prev_desired_ph != node_desired_ph['OWL322']:
                self.desired_ph_change_count += 1
                if self.prev_desired_ph in self.desired_phase_duration_dict.keys():
                    self.desired_phase_duration_dict[self.prev_desired_ph].append(self.desired_elapsed_time)
                else:
                    self.desired_phase_duration_dict[self.prev_desired_ph] = [self.desired_elapsed_time]
                self.desired_elapsed_time = 0
            else:
                self.desired_elapsed_time += 1
            self.prev_desired_ph = node_desired_ph['OWL322']

            if self.prev_actual_ph is not None and self.prev_actual_ph != self.nodes['OWL322'].cur_phase_id:
                self.actual_ph_change_count += 1
                if self.prev_actual_ph in self.actual_phase_duration_dict.keys():
                    self.actual_phase_duration_dict[self.prev_actual_ph].append(self.elapsed_time)
                else:
                    self.actual_phase_duration_dict[self.prev_actual_ph] = [self.elapsed_time]
                self.elapsed_time = 0
            else:
                self.elapsed_time += 1
            self.prev_actual_ph = self.nodes['OWL322'].cur_phase_id

            if ki_run == 'T':
                self.ki_run = 1
                self.ki_run_count += 1
            else:
                self.ki_run = 0

        if self.analysis_logs and self.env_config['env'] == 'owl322' and self.algo != 'adaptive':
            logger.info(f"{self.cur_interactionsteps+54000}- The desired phase for node OWL322 is : {node_desired_ph['OWL322']}")
            logger.info(f"{self.cur_interactionsteps+54000}- The actual phase for node OWL322 is : {self.nodes['OWL322'].cur_phase_id}")
            logger.info(f"{self.cur_interactionsteps+54000}- KI_Run: {ki_run}, self.KI_Run: {self.ki_run},"
                        f" elapsed_time: {self.elapsed_time}, desired_elapsed_time: {self.desired_elapsed_time}")
            logger.info(f"{self.cur_interactionsteps+54000}- cum_waits: S:{cum_waits[0]+cum_waits[1]+cum_waits[2]},"
                        f" N: {cum_waits[3]+cum_waits[4]+cum_waits[5]},"
                        f" W: {cum_waits[6]+cum_waits[7]+cum_waits[8]},"
                        f" E: {cum_waits[9]+cum_waits[10]+cum_waits[11]}")
            logger.info(f"{self.cur_interactionsteps+54000}- cum_waits: W_St: {cum_waits[7]}, W_L: {cum_waits[8]}")
            logger.info(f"{self.cur_interactionsteps+54000}- waves: S:{waves[0] + waves[1] + waves[2]},"
                        f" N: {waves[3] + waves[4] + waves[5]},"
                        f" W: {waves[6] + waves[7] + waves[8]},"
                        f" E: {waves[9] + waves[10] + waves[11]}")
            logger.info(f"{self.cur_interactionsteps + 54000}- waves: W_St: {waves[7]}, W_L: {waves[8]}")
            logger.info(f"{self.cur_interactionsteps+54000}- queue_lengths: S:{queue_lengths[0] + queue_lengths[1] + queue_lengths[2]},"
                        f" N: {queue_lengths[3] + queue_lengths[4] + queue_lengths[5]},"
                        f" W: {queue_lengths[6] + queue_lengths[7] + queue_lengths[8]},"
                        f" E: {queue_lengths[9] + queue_lengths[10] + queue_lengths[11]}")
            logger.info(f"{self.cur_interactionsteps + 54000}- queue_lengths: W_St: {queue_lengths[7]}, W_L: {queue_lengths[8]}")
            logger.info(f"{self.cur_interactionsteps+54000}- cur_avg_speed: S:{cur_avg_speed[0] + cur_avg_speed[1] + cur_avg_speed[2]},"
                        f" N: {cur_avg_speed[3] + cur_avg_speed[4] + cur_avg_speed[5]},"
                        f" W: {cur_avg_speed[6] + cur_avg_speed[7] + cur_avg_speed[8]},"
                        f" E: {cur_avg_speed[9] + cur_avg_speed[10] + cur_avg_speed[11]}")
            logger.info(f"{self.cur_interactionsteps + 54000}- cur_avg_speed: W_St: {cur_avg_speed[7]}, W_L: {cur_avg_speed[8]}")
            logger.info(f"{self.cur_interactionsteps+54000}- ped_wait_array: NS: {ped_wait_array[0]}, EW: {ped_wait_array[1]}")
        reward = self._calc_reward()
        self.cur_interactionsteps += 1
        return state, reward, done, dict(waves=waves, cum_waits=cum_waits, queues=queue_lengths, cur_phase_id=phase_id,
                                         cur_avg_speed=cur_avg_speed, desired_phase=desire_phase_id,
                                         ns_pedestrian_wait_time=ped_wait_array[0],
                                         ew_pedestrian_wait_time=ped_wait_array[1])

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

                phaseid_one_hot_encoded = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                phaseid_one_hot_encoded[self.nodes[node_name].cur_phase_id - 1] = 1

                desired_ph_one_hot_encoded = np.array([0, 0, 0, 0, 0, 0, 0, 0])
                desired_ph_one_hot_encoded[self.nodes[node_name].desired_phase_id - 1] = 1

                if count == 0:
                    phaseid_one_hot_encoded_arr = phaseid_one_hot_encoded
                    pedestrian_demand_arr = [(np.array(self.nodes[node_name].ns_pedestrian_wait_time) /
                                              self.mdp_dict['ped_wait_norm']).clip(min=0, max=2),
                                             (np.array(self.nodes[node_name].ew_pedestrian_wait_time) /
                                              self.mdp_dict['ped_wait_norm']).clip(min=0, max=2)]
                else:
                    phaseid_one_hot_encoded_arr = np.concatenate(
                        ([phaseid_one_hot_encoded_arr, phaseid_one_hot_encoded]))
                    pedestrian_demand_arr = np.concatenate(
                        (pedestrian_demand_arr, [(np.array(self.nodes[node_name].ns_pedestrian_wait_time) /
                                                  self.mdp_dict['ped_wait_norm']).clip(min=0, max=2),
                                                 (np.array(self.nodes[node_name].ew_pedestrian_wait_time) /
                                                  self.mdp_dict['ped_wait_norm']).clip(min=0, max=2)]))

            else:
                phaseid_one_hot_encoded = np.array([0, 0])
                desired_ph_one_hot_encoded = np.array([0, 0])
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

        return np.concatenate([cum_waits_arr, waves_arr, queue_lengths_arr, avg_speed_arr,
                               phaseid_one_hot_encoded_arr, pedestrian_demand_arr, desired_ph_one_hot_encoded,
                               [self.elapsed_time/self.mdp_dict['elapsed_norm']], [self.desired_elapsed_time/self.mdp_dict['elapsed_norm']]]), waves_arr, \
            cum_waits_arr, queue_lengths_arr, phaseid_one_hot_encoded_arr, avg_speed_arr, pedestrian_demand_arr, desired_ph_one_hot_encoded

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

            cum_wait = np.array(self.nodes[node_name].cur_cum_wait).sum() / self.nodes[node_name]. \
                len_lanes_in / self.mdp_dict['wait_norm']

            ns_ped_wait_time = self.nodes[node_name].ns_pedestrian_wait_time
            ew_ped_wait_time = self.nodes[node_name].ew_pedestrian_wait_time
            cum_ped_wait_time = (ns_ped_wait_time + ew_ped_wait_time)/self.mdp_dict['ped_wait_norm']
            reward = -1 * (queue + self.mdp_dict['wait_veh_reward_coef'] * cum_wait
                           + self.mdp_dict['wait_ped_reward_coef'] * cum_ped_wait_time)
            reward = reward.clip(min=-3, max=3)
            self.nodes[node_name].cur_reward = reward
            self.nodes[node_name].cum_reward += reward
            if self.analysis_logs:
                logger.info(f"{self.cur_interactionsteps+54000}- queue: {queue},"
                            f" cum_wait: {cum_wait} * {self.mdp_dict['wait_veh_reward_coef']},"
                            f" cum_ped_wait: {cum_ped_wait_time} * {self.mdp_dict['wait_ped_reward_coef']}")
                logger.info(f"{self.cur_interactionsteps+54000}- Reward after clipping: {reward}")
        return reward

    def getQueueLengthsNodeMean(self):
        queue_length_nodes = []
        for node_name in self.node_names:
            queue_lengths = self.nodes[node_name].cur_queue
            queue_lengths_mean = np.sum(queue_lengths)
            queue_length_nodes.append(queue_lengths_mean)
        queue_length_nodes_mean = np.mean(queue_length_nodes)
        return queue_length_nodes_mean

    def getWaitTimeNodeMean(self):
        wait_time_nodes = []
        for node_name in self.node_names:
            wait_time = self.nodes[node_name].cur_wait
            wait_time_mean = np.sum(wait_time)
            wait_time_nodes.append(wait_time_mean)
        wait_time_nodes_mean = np.mean(wait_time_nodes)
        return wait_time_nodes_mean

    def getSpeedNodeMean(self):
        speed_nodes = []
        for node_name in self.node_names:
            speed = self.nodes[node_name].cur_avg_speed
            speed_filtered = list(filter(lambda x: x != -1, speed))
            if len(speed_filtered) > 0:
                speed_mean = np.sum(speed_filtered)
            else:
                speed_mean = 0
            speed_nodes.append(speed_mean)
        speed_nodes_mean = np.mean(speed_nodes)
        return speed_nodes_mean

    def getPedestrianWaitNodeMean(self):
        pedestrian_wait_nodes = []
        for node_name in self.node_names:
            pedestrian_wait = self.nodes[node_name].ns_pedestrian_wait_time + self.nodes[node_name].ew_pedestrian_wait_time
            pedestrian_wait_nodes.append(pedestrian_wait)
        pedestrian_wait_nodes_mean = np.mean(pedestrian_wait_nodes)
        return pedestrian_wait_nodes_mean

    def getRewardMean(self):
        reward_nodes = []
        for node_name in self.node_names:
            reward = self.nodes[node_name].cur_reward
            reward_nodes.append(reward)
        reward_mean = np.mean(reward_nodes)
        return reward_mean

    def getCumRewardMean(self):
        cum_reward_nodes = []
        for node_name in self.node_names:
            cum_reward = self.nodes[node_name].cum_reward
            cum_reward_nodes.append(cum_reward)
        cum_reward_mean = np.mean(cum_reward_nodes)
        return cum_reward_mean

    def getPhaseCounts(self):
        return self.desired_ph_list, self.actual_ph_list

    def getPhaseChangeCounts(self):
        return self.desired_ph_change_count, self.actual_ph_change_count, self.ki_run_count

    def getActualPhaseDurationDict(self):
        return self.actual_phase_duration_dict

    def getDesiredPhaseDurationDict(self):
        return self.desired_phase_duration_dict

    def __get_detector_str(self, ns_time, ew_time):
        count = 0
        det_str = ''
        for det, det_type in self.detector_type_dict.items():
            num_vehicles = None
            if det_type == 'e1':
                num_vehicles = traci.inductionloop.getLastStepVehicleNumber(det)
            elif det_type == 'e2':
                num_vehicles = traci.lanearea.getLastStepVehicleNumber(det)
            else:
                num_vehicles = traci.multientryexit.getLastStepVehicleNumber(det)
            if num_vehicles > 0:
                if det_str != '':
                    det_str += '/'
                if count > 0:
                    det_str += f'({count})'
                det_str += f'{num_vehicles}'
                self.active_det_dict[det] = num_vehicles
            elif det in self.active_det_dict.keys():
                if det_str != '':
                    det_str += '/'
                if count > 0:
                    det_str += f'({count})'
                det_str += f'-{self.active_det_dict[det]}'
                del self.active_det_dict[det]
            count += 1
        if ns_time > 0:
            if det_str != '':
                det_str += '/'
            det_str += f'(12)1/(20)1'
            self.active_det_dict['ns'] = 1
        elif 'ns' in self.active_det_dict.keys():
            if det_str != '':
                det_str += '/'
            det_str += f'(12)-1/(20)-1'
            del self.active_det_dict['ns']
        if ew_time > 0:
            if det_str != '':
                det_str += '/'
            det_str += f'(16)1/(24)1'
            self.active_det_dict['ew'] = 1
        elif 'ew' in self.active_det_dict.keys():
            if det_str != '':
                det_str += '/'
            det_str += f'(16)-1/(24)-1'
            del self.active_det_dict['ew']

        if det_str != '':
            det_str += '/'
        det_str += '(29)1/(30)1'
        return det_str

    @staticmethod
    def _get_ped_wait_time(person_dict):
        if len(person_dict) > 0:
            sorted_person_dict = dict(sorted(person_dict.items(), key=lambda item:item[1], reverse=True))
            return next(iter(sorted_person_dict.values()))
        else:
            return 0
