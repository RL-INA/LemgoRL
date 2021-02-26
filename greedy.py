"""
Greedy Policy for Lemgo environment
@author: Vishal Rangras (Fraunhofer IOSB-INA in Lemgo)
@email: vishal.rangras@iosb-ina.fraunhofer.de
"""

import logging
import time
from pathlib import Path


MAX_GREENTIME = 90
MIN_WAVE = 8/28  # 0.2857

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


class LongestQueueFirstPolicyLemgoWvc:

    def __init__(self, env_config):

        self.controlled_nodes = sorted(env_config['controlled_nodes'])
        self.node_phase_demand_dict = {x: {'N_S': 0, 'E_W': 0} for x in self.controlled_nodes}
        self.flawed = False
        self.pedestrian_control = True
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

    def _determine_cur_phase(self, state):
        phase_dict = {}
        controlled_node_len = len(self.controlled_nodes)
        subtraction_val = 12
        for controlled_node in self.controlled_nodes:
            curr_phase = 0
            for i in range(len(state)-(controlled_node_len*subtraction_val), len(state)):
                if state[i] == 1:
                    # logger.info('Controlled Node: {} - Current Phase: {}'.format(controlled_node, curr_phase))
                    break
                curr_phase += 1
            controlled_node_len -= 1
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
        phase_dict = self._determine_cur_phase(state)
        logger.debug('Queue length and Wave Dict:')
        logger.debug(node_metric_dict)
        actions = {}

        logger.debug('Green Times before computing actions:')
        logger.debug(self.greentime)
        controlled_node_len = len(self.controlled_nodes)
        subtraction_val = 2
        for controlled_node in self.controlled_nodes:
            ns_pedestrian = state[len(state) - (controlled_node_len*subtraction_val)]
            ew_pedestrian = state[len(state) - (controlled_node_len*subtraction_val) + 1]
            self.greentime[controlled_node] += 1
            curr_phase = phase_dict[controlled_node]
            new_phase = curr_phase
            queue_length_dict = node_metric_dict[controlled_node + '_queue_dict']
            wave_length_dict = node_metric_dict[controlled_node + '_wave_dict']

            # Phase 1 is all Red. Phase 3 or Phase 7 kick starts the traffic lights.

            if curr_phase == 1 and self.flawed:
                if wave_length_dict[7] > MIN_WAVE:
                    new_phase = 7
                else:
                    new_phase = 3
                self.greentime[controlled_node] = 0

            elif curr_phase == 1:
                if self.node_phase_demand_dict[controlled_node]['N_S'] > \
                        self.node_phase_demand_dict[controlled_node]['E_W'] or wave_length_dict[7] > MIN_WAVE:
                    new_phase = 7
                else:
                    new_phase = 3
                self.greentime[controlled_node] = 0

            # Longest Wave first policy implemented below
            elif wave_length_dict[curr_phase] < MIN_WAVE or self.greentime[controlled_node] > MAX_GREENTIME:
                if curr_phase == 2 or curr_phase == 3 or curr_phase == 4:
                    self.node_phase_demand_dict[controlled_node]['E_W'] = 0
                    if wave_length_dict[4] > MIN_WAVE:
                        new_phase = 4
                    else:
                        new_phase = 7
                        self.node_phase_demand_dict[controlled_node]['N_S'] += 1
                        self.greentime[controlled_node] = 0
                elif curr_phase == 6 or curr_phase == 7 or curr_phase == 8:
                    self.node_phase_demand_dict[controlled_node]['N_S'] = 0
                    if wave_length_dict[8] > MIN_WAVE:
                        new_phase = 8
                    else:
                        new_phase = 3
                        self.node_phase_demand_dict[controlled_node]['E_W'] += 1
                        self.greentime[controlled_node] = 0

            if self.pedestrian_control:
                if new_phase == 7 and ns_pedestrian > 0:
                    new_phase = 6
                elif new_phase == 3 and ew_pedestrian > 0:
                    new_phase = 2

            logger.debug(f"N_S Ped: {ns_pedestrian}")
            logger.debug(f"E_W Ped: {ew_pedestrian}")

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
