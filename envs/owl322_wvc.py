"""
OWL322 Environment - TrafficSimulationBase extended Class
This class is used when Lisa+ Virtual Controller is needed in the loop.
@author: Vishal Rangras (Fraunhofer IOSB-INA in Lemgo)
@email: vishal.rangras@iosb-ina.fraunhofer.de
"""
from envs.env_wvc import TrafficSimulatorWvcBase

MAP_ACTION_TO_DESIREDPHASE = {0: 0, 1: 2, 2: 4, 3: 6}
MAP_NEXTPHASE_TO_YELLOWPHASE = {0: 7, 2: 1, 4: 3, 6: 5}


class TrafficSimulation(TrafficSimulatorWvcBase):
    def __init__(self, env_config):
        sumo_dict = env_config['sumo_dict']
        mdp_dict = env_config['mdp_dict']
        sumo_env_config_dir = env_config['sumo_env_config_dir']
        gui = env_config['gui']
        sumo_connector = env_config['sumo_connector']
        skip_worker = env_config['skip_worker']
        env_config['action_phase_map'] = MAP_ACTION_TO_DESIREDPHASE
        env_config['yellow_phase_map'] = MAP_NEXTPHASE_TO_YELLOWPHASE
        print_metrics = env_config['print_metrics']
        if skip_worker:
            super().__init__(sumo_dict, mdp_dict, sumo_env_config_dir, gui, sumo_connector,
                             env_config, 0, print_metrics=print_metrics)
        else:
            super().__init__(sumo_dict, mdp_dict, sumo_env_config_dir, gui, sumo_connector,
                             env_config, worker_index=env_config.worker_index, print_metrics=print_metrics)
