"""
Callback functions script for Ray custom metrics.
@author: Arthur Müller (Fraunhofer IOSB-INA in Lemgo)
@email: arthur.mueller@iosb-ina.fraunhofer.de
"""

from typing import Dict
import numpy as np
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        """
        Callback run on the rollout worker before each episode starts.

                Args:
                    worker (RolloutWorker): Reference to the current rollout worker.
                    base_env (BaseEnv): BaseEnv running the episode. The underlying
                        env object can be gotten by calling base_env.get_unwrapped().
                    policies (dict): Mapping of policy id to policy objects. In single
                        agent mode there will only be a single "default" policy.
                    episode (MultiAgentEpisode): Episode object which contains episode
                        state. You can use the `episode.user_data` dict to store
                        temporary data, and `episode.custom_metrics` to store custom
                        metrics for the episode.
                    kwargs: Forward compatibility placeholder.
        """
        episode.user_data["queue_length"] = []
        episode.user_data["cum_wait"] = []
        episode.user_data["wave"] = []
        episode.user_data["cur_avg_speed"] = []
        # episode.user_data["cur_phase_id"] = []
        # episode.user_data["desired_phase"] = []
        episode.user_data["ns_pedestrian_wait_time"] = []
        episode.user_data["ew_pedestrian_wait_time"] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        """
        Runs on each episode step.

                Args:
                    worker (RolloutWorker): Reference to the current rollout worker.
                    base_env (BaseEnv): BaseEnv running the episode. The underlying
                        env object can be gotten by calling base_env.get_unwrapped().
                    episode (MultiAgentEpisode): Episode object which contains episode
                        state. You can use the `episode.user_data` dict to store
                        temporary data, and `episode.custom_metrics` to store custom
                        metrics for the episode.
                    kwargs: Forward compatibility placeholder.
        """
        if not isinstance(episode.last_info_for('agent0'), type(None)):
            queue_length = episode.last_info_for('agent0')['queues']
            wave = episode.last_info_for('agent0')['waves']
            cum_wait = episode.last_info_for('agent0')['cum_waits']
            cur_avg_speed = episode.last_info_for('agent0')['cur_avg_speed']
            # cur_phase_id = episode.last_info_for('agent0')['cur_phase_id']
            # desired_phase = episode.last_info_for('agent0')['desired_phase']
            ns_pedestrian_wait_time = episode.last_info_for('agent0')['ns_pedestrian_wait_time']
            ew_pedestrian_wait_time = episode.last_info_for('agent0')['ew_pedestrian_wait_time']

            episode.user_data["queue_length"].append(queue_length)
            episode.user_data["cum_wait"].append(cum_wait)
            episode.user_data["wave"].append(wave)
            episode.user_data["cur_avg_speed"].append(cur_avg_speed)
            # episode.user_data["cur_phase_id"].append(cur_phase_id)
            # episode.user_data["desired_phase"].append(desired_phase)
            episode.user_data["ns_pedestrian_wait_time"].append(ns_pedestrian_wait_time)
            episode.user_data["ew_pedestrian_wait_time"].append(ew_pedestrian_wait_time)

    def on_episode_end(self,  worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        """
        Runs when an episode is done.

                Args:
                    worker (RolloutWorker): Reference to the current rollout worker.
                    base_env (BaseEnv): BaseEnv running the episode. The underlying
                        env object can be gotten by calling base_env.get_unwrapped().
                    policies (dict): Mapping of policy id to policy objects. In single
                        agent mode there will only be a single "default" policy.
                    episode (MultiAgentEpisode): Episode object which contains episode
                        state. You can use the `episode.user_data` dict to store
                        temporary data, and `episode.custom_metrics` to store custom
                        metrics for the episode.
                    kwargs: Forward compatibility placeholder.
        """
        episode.custom_metrics["queue_length"] = np.mean(episode.user_data["queue_length"])
        episode.custom_metrics["cum_wait"] = np.mean(episode.user_data["cum_wait"])
        episode.custom_metrics["wave"] = np.mean(episode.user_data["wave"])
        episode.custom_metrics["cur_avg_speed"] = np.mean(episode.user_data["cur_avg_speed"])
        # episode.custom_metrics["cur_phase_id"] = episode.user_data["cur_phase_id"]
        # episode.custom_metrics["desired_phase"] = episode.user_data["desired_phase"]
        episode.custom_metrics["ns_pedestrian_wait_time"] = np.mean(episode.user_data["ns_pedestrian_wait_time"])
        episode.custom_metrics["ew_pedestrian_wait_time"] = np.mean(episode.user_data["ew_pedestrian_wait_time"])


    # def on_train_result(self, trainer, result: dict, **kwargs):
    #     """Called at the end of Trainable.train().
    #
    #     Args:
    #         trainer (Trainer): Current trainer instance.
    #         result (dict): Dict of results returned from trainer.train() call.
    #             You can mutate this object to add additional metrics.
    #         kwargs: Forward compatibility placeholder.
    #     """

    # def on_postprocess_trajectory(
    #         self, worker: RolloutWorker, episode: MultiAgentEpisode,
    #         agent_id: AgentID, policy_id: PolicyID,
    #         policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
    #         original_batches: Dict[AgentID, SampleBatch], **kwargs):
    #     """Called immediately after a policy's postprocess_fn is called.
    #
    #     You can use this callback to do additional postprocessing for a policy,
    #     including looking at the trajectory data of other agents in multi-agent
    #     settings.
    #
    #     Args:
    #         worker (RolloutWorker): Reference to the current rollout worker.
    #         episode (MultiAgentEpisode): Episode object.
    #         agent_id (str): Id of the current agent.
    #         policy_id (str): Id of the current policy for the agent.
    #         policies (dict): Mapping of policy id to policy objects. In single
    #             agent mode there will only be a single "default" policy.
    #         postprocessed_batch (SampleBatch): The postprocessed sample batch
    #             for this agent. You can mutate this object to apply your own
    #             trajectory postprocessing.
    #         original_batches (dict): Mapping of agents to their unpostprocessed
    #             trajectory data. You should not mutate this object.
    #         kwargs: Forward compatibility placeholder.
    #     """
