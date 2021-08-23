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
Greedy Policy for Lemgo environment
@author: Vishal Rangras (Fraunhofer IOSB-INA in Lemgo)
@email: vishal.rangras@iosb-ina.fraunhofer.de
"""

import logging
import time
from pathlib import Path
import numpy as np


MIN_GREENTIME = 10
MAX_GREENTIME = 90
MIN_WAVE = 8/28  # 0.2857
MIN_QUEUE = 8/28

MAP_PHASE_TO_ACTION = {0: 0, 2: 1, 4: 2, 6: 3}


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-greedy-'+timestamp+'.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()


class LongestWaveFirstPolicyLemgoWvc:

    def __init__(self, env_config):

        self.controlled_nodes = sorted(env_config['controlled_nodes'])
        self.node_phase_demand_dict = {x: {'N_S': 0, 'E_W': 0} for x in self.controlled_nodes}
        self.flawed = False
        self.pedestrian_control = True
        self.old_logic = False
        self.curr_ph_list = []
        self.next_ph_list = []
        self.wave_e = []
        self.wave_w = []
        self.wave_s = []
        self.wave_n = []
        if len(self.controlled_nodes) == 2:
            self.greentime = {'OWL322': 0, 'Bi308': 0}
        else:
            if 'OWL322' in self.controlled_nodes:
                self.greentime = {'OWL322': 0}
            elif 'Bi308' in self.controlled_nodes:
                self.greentime = {'Bi308': 0}
        self.state_config = env_config['state_space_config']

    def _calc_queue_length_and_wave(self, state) -> dict:

        que_s = self.state_config['queue_start_ind']
        wav_s = self.state_config['wave_start_ind']

        node_metric_dict = {}

        count = 0

        for controlled_node in self.controlled_nodes:

            if count == 1:
                que_s += 12
                wav_s += 12

            if controlled_node == 'OWL322':
                phase_2_ql = state[que_s+9] + state[que_s+10] + state[que_s+11]\
                             + state[que_s+6] + state[que_s+7] + state[que_s+8]
                phase_3_ql = state[que_s+9] + state[que_s+10] + state[que_s+11]\
                    + state[que_s+6] + state[que_s+7] + state[que_s+8]
                phase_4_ql = state[que_s+6] + state[que_s+7] + state[que_s+8]
                phase_6_ql = state[que_s+3] + state[que_s+4] + state[que_s+5]\
                    + state[que_s+0] + state[que_s+1] + state[que_s+2]
                phase_7_ql = state[que_s + 3] + state[que_s + 4] + state[que_s + 5]\
                    + state[que_s + 0] + state[que_s + 1] + state[que_s + 2]
                phase_8_ql = state[que_s + 3] + state[que_s + 4] + state[que_s + 5]

                phase_2_wave = state[wav_s+9] + state[wav_s+10] + state[wav_s+11]\
                    + state[wav_s+6] + state[wav_s+7] + state[wav_s+8]
                phase_3_wave = state[wav_s+9] + state[wav_s+10] + state[wav_s+11]\
                    + state[wav_s+6] + state[wav_s+7] + state[wav_s+8]
                phase_4_wave = state[wav_s+6] + state[wav_s+7] + state[wav_s+8]
                phase_6_wave = state[wav_s+3] + state[wav_s+4] + state[wav_s+5]\
                    + state[wav_s+0] + state[wav_s+1] + state[wav_s+2]
                phase_7_wave = state[wav_s+3] + state[wav_s+4] + state[wav_s+5]\
                    + state[wav_s+0] + state[wav_s+1] + state[wav_s+2]
                phase_8_wave = state[wav_s+3] + state[wav_s+4] + state[wav_s+5]

                queue_length_dict = {
                    2: phase_2_ql,
                    3: phase_3_ql,
                    4: phase_4_ql,
                    6: phase_6_ql,
                    7: phase_7_ql,
                    8: phase_8_ql
                }

                wave_dict = {
                    2: phase_2_wave,
                    3: phase_3_wave,
                    4: phase_4_wave,
                    6: phase_6_wave,
                    7: phase_7_wave,
                    8: phase_8_wave
                }

            elif controlled_node == 'Bi308':
                phase_0_ql = state[que_s+0] + state[que_s+1] + state[que_s+2] + state[que_s+9] + state[que_s+10]
                phase_2_ql = state[que_s+11] + state[que_s+9]
                phase_4_ql = state[que_s+6] + state[que_s+7] + state[que_s+3] + state[que_s+4]
                phase_6_ql = state[que_s+8] + state[que_s+5]

                phase_0_wave = state[wav_s+0] + state[wav_s+1] + state[wav_s+2] + state[wav_s+9] + state[wav_s+10]
                phase_2_wave = state[wav_s+11] + state[wav_s+9]
                phase_4_wave = state[wav_s+6] + state[wav_s+7] + state[wav_s+3] + state[wav_s+4]
                phase_6_wave = state[wav_s+8] + state[wav_s+5]

                queue_length_dict = {
                    0: phase_0_ql,
                    1: phase_2_ql,
                    2: phase_4_ql,
                    3: phase_6_ql
                }

                wave_dict = {
                    0: phase_0_wave,
                    1: phase_2_wave,
                    2: phase_4_wave,
                    3: phase_6_wave
                }

            node_metric_dict[controlled_node + '_queue_dict'] = queue_length_dict
            node_metric_dict[controlled_node + '_wave_dict'] = wave_dict

            count += 1

        return node_metric_dict

    def _calc_wave(self, state):
        wav_s = self.state_config['wave_start_ind']

        node_wave_metric = {}

        for controlled_node in self.controlled_nodes:
            if controlled_node == 'OWL322':

                south_wave = state[wav_s + 0] + state[wav_s + 1] + state[wav_s + 2]
                north_wave = state[wav_s + 3] + state[wav_s + 4] + state[wav_s + 5]
                west_wave = state[wav_s + 6] + state[wav_s + 7] + state[wav_s + 8]
                west_left_wave = state[wav_s + 8]
                east_wave = state[wav_s + 9] + state[wav_s + 10] + state[wav_s + 11]

                wave_dict = {
                    's': south_wave,
                    'n': north_wave,
                    'w': west_wave,
                    'wl': west_left_wave,
                    'e': east_wave
                }

                node_wave_metric[controlled_node + '_wave_dict'] = wave_dict
        return node_wave_metric

    def _determine_cur_phase(self, state):
        phase_dict = {}
        controlled_node_len = len(self.controlled_nodes)
        subtraction_val = 20
        for controlled_node in self.controlled_nodes:
            curr_phase = 0
            for i in range(len(state)-(controlled_node_len*subtraction_val), len(state)):
                if state[i] == 1:
                    break
                curr_phase += 1
            controlled_node_len -= 1
            curr_phase += 1  # Action Space to Phase Space mapping
            phase_dict[controlled_node] = curr_phase
        return phase_dict

    def compute_action(self, state):
        """
        Args:
            state: current state of the environment

        Returns:
            int: Computed Action from the action_space {2, 3, 4, 5, 6, 7, 8} for single node
            dict: Computed Actions from the action space {2, 3, 4, 5, 6, 7, 8} for both the nodes

        Note:
            Phase/(s) returned by this method are sent as AP Value KiWPh (KI Phase Wish) to
            Lisa+ virtual controller without any conversion or mapping.

        """
        node_metric_dict = self._calc_queue_length_and_wave(state)
        node_wave_metric_dict = self._calc_wave(state)
        phase_dict = self._determine_cur_phase(state)
        actions = {}

        controlled_node_len = len(self.controlled_nodes)
        subtraction_val = 12
        for controlled_node in self.controlled_nodes:
            ns_pedestrian = state[len(state) - (controlled_node_len*subtraction_val)]
            ew_pedestrian = state[len(state) - (controlled_node_len*subtraction_val) + 1]
            self.greentime[controlled_node] += 1
            curr_phase = phase_dict[controlled_node]
            self.curr_ph_list.append(curr_phase)
            new_phase = curr_phase
            queue_length_dict = node_metric_dict[controlled_node + '_queue_dict']
            wave_length_dict = node_metric_dict[controlled_node + '_wave_dict']
            wave_dict = node_wave_metric_dict[controlled_node + '_wave_dict']
            self.wave_e.append(wave_dict['e'])
            self.wave_w.append(wave_dict['w'])
            self.wave_n.append(wave_dict['n'])
            self.wave_s.append(wave_dict['s'])

            # Phase 1 is all Red. Phase 3 or Phase 7 kick starts the traffic lights.

            if curr_phase == 1 and self.flawed:
                if wave_length_dict[7] > MIN_WAVE:
                    new_phase = 7
                else:
                    new_phase = 3
                self.greentime[controlled_node] = 0

            elif curr_phase == 1 and self.old_logic:
                if self.node_phase_demand_dict[controlled_node]['N_S'] > \
                        self.node_phase_demand_dict[controlled_node]['E_W'] or wave_length_dict[7] > MIN_WAVE:
                    new_phase = 7
                else:
                    new_phase = 3
                self.greentime[controlled_node] = 0

            elif curr_phase == 1:
                if self.node_phase_demand_dict[controlled_node]['N_S'] > \
                        self.node_phase_demand_dict[controlled_node]['E_W']\
                        or (wave_dict['n'] + wave_dict['s']) > MIN_WAVE:
                    new_phase = 7
                else:
                    new_phase = 3
                self.greentime[controlled_node] = 0

            # Longest Wave first policy implemented below
            else:
                if self.old_logic:
                    if curr_phase in [2, 3, 4, 5]:
                        curr_dir_wave_length = wave_length_dict[3] + wave_length_dict[4]
                        orthogonal_wave_length = wave_length_dict[7] + wave_length_dict[8]
                        self.node_phase_demand_dict[controlled_node]['E_W'] = 0
                        if curr_dir_wave_length < orthogonal_wave_length:
                            new_phase = 7
                            self.node_phase_demand_dict[controlled_node]['N_S'] += 1
                            self.greentime[controlled_node] = 0
                        elif wave_length_dict[curr_phase] < wave_length_dict[4]:
                            new_phase = 4
                        else:
                            new_phase = 3
                    else:
                        curr_dir_wave_length = wave_length_dict[7] + wave_length_dict[8]
                        orthogonal_wave_length = wave_length_dict[3] + wave_length_dict[4]
                        self.node_phase_demand_dict[controlled_node]['N_S'] = 0
                        if curr_dir_wave_length < orthogonal_wave_length:
                            new_phase = 3
                            self.node_phase_demand_dict[controlled_node]['E_W'] += 1
                            self.greentime[controlled_node] = 0
                        elif wave_length_dict[curr_phase] < wave_length_dict[8]:
                            new_phase = 8
                        else:
                            new_phase = 7
                else:
                    if curr_phase == 2 or curr_phase == 3:
                        self.node_phase_demand_dict[controlled_node]['E_W'] = 0
                        curr_ph_wave_length = wave_dict['e'] + wave_dict['w']
                        orth_ph_wave_length = wave_dict['n'] + wave_dict['s']
                        if curr_ph_wave_length < orth_ph_wave_length and wave_dict['e'] < MIN_WAVE:
                            if wave_dict['wl'] > MIN_WAVE/3:
                                new_phase = 4
                            else:
                                new_phase = 7
                                self.node_phase_demand_dict[controlled_node]['N_S'] += 1
                                self.greentime[controlled_node] = 0
                        else:
                            new_phase = 3
                    elif curr_phase == 4:
                        self.node_phase_demand_dict[controlled_node]['E_W'] = 0
                        if wave_dict['wl'] > MIN_WAVE/3:
                            new_phase = 4
                        else:
                            new_phase = 7
                            self.node_phase_demand_dict[controlled_node]['N_S'] += 1
                            self.greentime[controlled_node] = 0
                    else:
                        curr_ph_wave_length = wave_dict['n'] + wave_dict['s']
                        orth_ph_wave_length = wave_dict['e'] + wave_dict['w']
                        self.node_phase_demand_dict[controlled_node]['N_S'] = 0
                        if curr_ph_wave_length < orth_ph_wave_length:
                            new_phase = 3
                            self.node_phase_demand_dict[controlled_node]['E_W'] += 1
                            self.greentime[controlled_node] = 0
                        elif wave_dict['n'] > wave_dict['s'] and wave_dict['s'] < MIN_WAVE:
                            new_phase = 8
                        else:
                            new_phase = curr_phase

            if self.pedestrian_control:
                if (new_phase == 7 and ns_pedestrian > 0) or (new_phase == 8 and ns_pedestrian > 10):
                    new_phase = 6
                elif new_phase == 3 and ew_pedestrian > 0:
                    new_phase = 2

            self.next_ph_list.append(new_phase)

            logger.debug(f"N_S Ped: {ns_pedestrian}")
            logger.debug(f"E_W Ped: {ew_pedestrian}")

            new_phase -= 1  # Phase Space to Action Space mapping
            actions[controlled_node] = new_phase
            logger.debug('Current Node: %s, Current Phase: %s, Computed next Phase: %s'
                         % (controlled_node, curr_phase, new_phase))
            logger.debug('Node Phase Demand of OWL322: %s' % self.node_phase_demand_dict['OWL322'])

        logger.debug('Green Times after computing actions:')
        logger.debug(self.greentime)
        if len(actions) == 1:
            return actions[self.controlled_nodes[0]]
        else:
            return actions

    def reset_measurements(self):
        self.curr_ph_list.clear()
        self.next_ph_list.clear()
        self.wave_e.clear()
        self.wave_w.clear()
        self.wave_n.clear()
        self.wave_s.clear()

    def get_measurements(self):
        return self.curr_ph_list, self.next_ph_list, self.wave_e, self.wave_w,\
               self.wave_n, self.wave_s


class LongestQueueFirstPolicyLemgoWvc:

    def __init__(self, env_config):

        self.controlled_nodes = sorted([env_config['controlled_nodes']])
        self.node_phase_demand_dict = {x: {'N_S': 0, 'E_W': 0} for x in self.controlled_nodes}
        self.pedestrian_control = True

        self.curr_ph_list = []
        self.next_ph_list = []
        if len(self.controlled_nodes) == 2:
            self.greentime = {'OWL322': 0, 'Bi308': 0}
        else:
            if 'OWL322' in self.controlled_nodes:
                self.greentime = {'OWL322': 0}
            elif 'Bi308' in self.controlled_nodes:
                self.greentime = {'Bi308': 0}
        self.state_config = env_config['state_space_config']

    def _calc_queue_length(self, state) -> dict:

        que_s = self.state_config['queue_start_ind']

        node_metric_dict = {}

        count = 0

        for controlled_node in self.controlled_nodes:

            if count == 1:
                que_s += 12

            if controlled_node == 'OWL322':
                phase_2_ql = state[que_s+9] + state[que_s+10] + state[que_s+11]\
                             + state[que_s+6] + state[que_s+7] + state[que_s+8]
                phase_3_ql = state[que_s+9] + state[que_s+10] + state[que_s+11]\
                    + state[que_s+6] + state[que_s+7] + state[que_s+8]
                phase_4_ql = state[que_s+6] + state[que_s+7] + state[que_s+8]
                phase_6_ql = state[que_s+3] + state[que_s+4] + state[que_s+5]\
                    + state[que_s+0] + state[que_s+1] + state[que_s+2]
                phase_7_ql = state[que_s + 3] + state[que_s + 4] + state[que_s + 5]\
                    + state[que_s + 0] + state[que_s + 1] + state[que_s + 2]
                phase_8_ql = state[que_s + 3] + state[que_s + 4] + state[que_s + 5]

                queue_length_dict = {
                    2: phase_2_ql,
                    3: phase_3_ql,
                    4: phase_4_ql,
                    6: phase_6_ql,
                    7: phase_7_ql,
                    8: phase_8_ql
                }

            elif controlled_node == 'Bi308':
                phase_0_ql = state[que_s+0] + state[que_s+1] + state[que_s+2] + state[que_s+9] + state[que_s+10]
                phase_2_ql = state[que_s+11] + state[que_s+9]
                phase_4_ql = state[que_s+6] + state[que_s+7] + state[que_s+3] + state[que_s+4]
                phase_6_ql = state[que_s+8] + state[que_s+5]

                queue_length_dict = {
                    0: phase_0_ql,
                    1: phase_2_ql,
                    2: phase_4_ql,
                    3: phase_6_ql
                }

            node_metric_dict[controlled_node + '_queue_dict'] = queue_length_dict

            count += 1

        return node_metric_dict

    def _calc_queue(self, state):
        que_s = self.state_config['queue_start_ind']

        node_queue_metric = {}

        for controlled_node in self.controlled_nodes:
            if controlled_node == 'OWL322':

                south_queue = state[que_s + 0] + state[que_s + 1] + state[que_s + 2]
                north_queue = state[que_s + 3] + state[que_s + 4] + state[que_s + 5]
                west_queue = state[que_s + 6] + state[que_s + 7] + state[que_s + 8]
                west_left_queue = state[que_s + 8]
                east_queue = state[que_s + 9] + state[que_s + 10] + state[que_s + 11]

                queue_dict = {
                    's': south_queue,
                    'n': north_queue,
                    'w': west_queue,
                    'wl': west_left_queue,
                    'e': east_queue
                }

                node_queue_metric[controlled_node + '_queue_dict'] = queue_dict
        return node_queue_metric

    def _determine_cur_phase(self, state):
        phase_dict = {}
        controlled_node_len = len(self.controlled_nodes)
        subtraction_val = 20
        for controlled_node in self.controlled_nodes:
            curr_phase = 0
            for i in range(len(state)-(controlled_node_len*subtraction_val), len(state)):
                if state[i] == 1:
                    break
                curr_phase += 1
            controlled_node_len -= 1
            curr_phase += 1  # Action Space to Phase Space mapping
            phase_dict[controlled_node] = curr_phase
        return phase_dict

    def compute_action(self, state):
        """
        Args:
            state: current state of the environment

        Returns:
            int: Computed Action from the action_space {2, 3, 4, 5, 6, 7, 8} for single node
            dict: Computed Actions from the action space {2, 3, 4, 5, 6, 7, 8} for both the nodes

        Note:
            Phase/(s) returned by this method are sent as AP Value KiWPh (KI Phase Wish) to
            Lisa+ virtual controller without any conversion or mapping.

        """
        node_metric_dict = self._calc_queue_length(state)
        node_queue_metric_dict = self._calc_queue(state)
        phase_dict = self._determine_cur_phase(state)
        logger.debug('Queue length Dict:')
        logger.debug(node_metric_dict)
        actions = {}

        logger.debug('Green Times before computing actions:')
        logger.debug(self.greentime)
        controlled_node_len = len(self.controlled_nodes)
        subtraction_val = 12
        for controlled_node in self.controlled_nodes:
            ns_pedestrian = state[len(state) - (controlled_node_len*subtraction_val)]
            ew_pedestrian = state[len(state) - (controlled_node_len*subtraction_val) + 1]
            self.greentime[controlled_node] += 1
            curr_phase = phase_dict[controlled_node]
            self.curr_ph_list.append(curr_phase)
            new_phase = curr_phase
            queue_length_dict = node_metric_dict[controlled_node + '_queue_dict']
            queue_dict = node_queue_metric_dict[controlled_node + '_queue_dict']

            # Phase 1 is all Red. Phase 3 or Phase 7 kick starts the traffic lights.

            if curr_phase == 1:
                if self.node_phase_demand_dict[controlled_node]['N_S'] > \
                        self.node_phase_demand_dict[controlled_node]['E_W']\
                        or (queue_dict['n'] + queue_dict['s']) > MIN_QUEUE:
                    new_phase = 7
                else:
                    new_phase = 3
                self.greentime[controlled_node] = 0

            # Longest Queue first policy implemented below
            else:
                if curr_phase == 2 or curr_phase == 3:
                    self.node_phase_demand_dict[controlled_node]['E_W'] = 0
                    curr_ph_queue_length = queue_dict['e'] + queue_dict['w']
                    orth_ph_queue_length = queue_dict['n'] + queue_dict['s']
                    if curr_ph_queue_length < orth_ph_queue_length and queue_dict['e'] < MIN_QUEUE:
                        if queue_dict['wl'] > MIN_QUEUE/3:
                            new_phase = 4
                        else:
                            new_phase = 7
                            self.node_phase_demand_dict[controlled_node]['N_S'] += 1
                            self.greentime[controlled_node] = 0
                    else:
                        new_phase = 3
                elif curr_phase == 4:
                    self.node_phase_demand_dict[controlled_node]['E_W'] = 0
                    if queue_dict['wl'] > MIN_QUEUE/3:
                        new_phase = 4
                    else:
                        new_phase = 7
                        self.node_phase_demand_dict[controlled_node]['N_S'] += 1
                        self.greentime[controlled_node] = 0
                else:
                    orth_ph_queue_length = queue_dict['e'] + queue_dict['w']
                    curr_ph_queue_length = queue_dict['n'] + queue_dict['s']
                    self.node_phase_demand_dict[controlled_node]['N_S'] = 0
                    if curr_ph_queue_length < orth_ph_queue_length:
                        new_phase = 3
                        self.node_phase_demand_dict[controlled_node]['E_W'] += 1
                        self.greentime[controlled_node] = 0
                    elif queue_dict['n'] > queue_dict['s'] and queue_dict['s'] < MIN_QUEUE:
                        new_phase = 8
                    else:
                        new_phase = curr_phase

            if self.pedestrian_control:
                if (new_phase == 7 and ns_pedestrian > 0) or (new_phase == 8 and ns_pedestrian > 10):
                    new_phase = 6
                elif new_phase == 3 and ew_pedestrian > 0:
                    new_phase = 2

            self.next_ph_list.append(new_phase)

            new_phase -= 1  # Phase Space to Action Space mapping
            actions[controlled_node] = new_phase

        if len(actions) == 1:
            return actions[self.controlled_nodes[0]]
        else:
            return actions


class FixedTimeWvc:

    def __init__(self, env_config):
        self.phase_duration = {2: 23, 3: 20, 4: 8, 5: 6, 6: 14, 7: 5, 8: 4}
        self.curr_to_next_ph = {2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 2}
        self.elapsed_time = 0
        self.curr_phase = 2

    def compute_action(self, state):
        if self.elapsed_time > self.phase_duration[self.curr_phase]:
            self.curr_phase = self.curr_to_next_ph[self.curr_phase]
            self.elapsed_time = 0
        else:
            self.elapsed_time += 1
        return self.curr_phase


class RandomTimeWvc:

    def __init__(self, env_config):
        self.curr_to_next_ph = {2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 2}
        self.elapsed_time = 0
        self.max_time = 0
        self.curr_phase = 2

    def compute_action(self, state):
        if self.elapsed_time > self.max_time:
            self.curr_phase = self.curr_to_next_ph[self.curr_phase]
            self.elapsed_time = 0
            self.max_time = np.random.randint(0, 50)
        else:
            self.elapsed_time += 1
        return self.curr_phase


class AdaptivePolicy:

    def __init__(self, env_config):
        self.env_config = env_config

    def compute_action(self, state):
        return 1
