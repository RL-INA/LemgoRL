import numpy as np
import pandas as pd
from utils import CustomPlot
import xlsxwriter as xl

t = np.arange(0., 4200.)


def plot_moving_average_over_episode(metric_episode_list, metric_name, metric_path,
                                     algo, checkpoint_nr, metric_avg_algos_list, metric_std_algos_list):
    ma_metric_episodes_list = []
    for orig_metric_list in metric_episode_list:
        ma_metric_episodes_list.append(pd.Series(orig_metric_list).rolling(100).mean().fillna(0).to_list())

    ma_metric_avg_over_ep = np.mean(ma_metric_episodes_list, axis=0)
    ma_metric_std_over_ep = np.std(ma_metric_episodes_list, axis=0)
    ma_std_err = ma_metric_std_over_ep / np.sqrt(len(ma_metric_episodes_list))
    ma_std_err *= 1.96
    CustomPlot.save_ci_plot(f'{metric_path}{metric_name}_Over_Episodes_{algo}_{checkpoint_nr}.png',
                            'Simulation time (s)', f'{metric_name}', f'{metric_name} over Timestep',
                            t, ma_metric_avg_over_ep, ma_metric_avg_over_ep - ma_std_err,
                            ma_metric_avg_over_ep + ma_std_err, algo)  # algo+'_'+checkpoint_nr

    metric_avg_algos_list.append(ma_metric_avg_over_ep)
    metric_std_algos_list.append(ma_std_err)
    return np.mean(ma_metric_avg_over_ep), np.std(ma_metric_avg_over_ep)


def compute_metric_mean_std(metric_episode_list):
    inter_mean = np.mean(metric_episode_list, axis=0)
    return np.mean(inter_mean), np.std(inter_mean)


def save_summary_table(result_path,
                       metric_summary_dict, ma=False):
    queue_algos = metric_summary_dict['queue']
    wait_time_algos = metric_summary_dict['wait_time']
    ped_wait_time_algos = metric_summary_dict['ped_wait_time']
    speed_algos = metric_summary_dict['speed']
    reward_algos = metric_summary_dict['reward']
    cum_reward_algos = metric_summary_dict['cum_reward']

    if ma:
        workbook = xl.Workbook(f'{result_path}/Summary_Report_MA.xlsx')
    else:
        workbook = xl.Workbook(f'{result_path}/Summary_Report.xlsx')
    worksheet = workbook.add_worksheet()
    row_count = 0
    col_count = 0
    worksheet.write(row_count, col_count + 1, 'Queue_Mean')
    worksheet.write(row_count, col_count + 2, 'Queue_Std')
    worksheet.write(row_count, col_count + 3, 'Wait_Time_Mean')
    worksheet.write(row_count, col_count + 4, 'Wait_Time_Std')
    worksheet.write(row_count, col_count + 5, 'Speed_Mean')
    worksheet.write(row_count, col_count + 6, 'Speed_Std')
    worksheet.write(row_count, col_count + 7, 'Ped_Wait_Time_Mean')
    worksheet.write(row_count, col_count + 8, 'Ped_Wait_Time_Std')
    worksheet.write(row_count, col_count + 9, 'Reward_Mean')
    worksheet.write(row_count, col_count + 10, 'Reward_Std')
    worksheet.write(row_count, col_count + 11, 'Cum_Reward_Mean')
    worksheet.write(row_count, col_count + 12, 'Cum_Reward_Std')

    for queue, wait, ped_wait, speed, reward, cum_reward in zip(queue_algos, wait_time_algos,
                                            ped_wait_time_algos, speed_algos,
                                            reward_algos, cum_reward_algos):

        row_count += 1
        worksheet.write(row_count, col_count, queue.algo)
        worksheet.write(row_count, col_count + 1, queue.mean)
        worksheet.write(row_count, col_count + 2, queue.std)
        worksheet.write(row_count, col_count + 3, wait.mean)
        worksheet.write(row_count, col_count + 4, wait.std)
        worksheet.write(row_count, col_count + 5, speed.mean)
        worksheet.write(row_count, col_count + 6, speed.std)
        worksheet.write(row_count, col_count + 7, ped_wait.mean)
        worksheet.write(row_count, col_count + 8, ped_wait.std)
        worksheet.write(row_count, col_count + 9, reward.mean)
        worksheet.write(row_count, col_count + 10, reward.std)
        worksheet.write(row_count, col_count + 11, cum_reward.mean)
        worksheet.write(row_count, col_count + 12, cum_reward.std)

    workbook.close()


class AlgoSummary:

    def __init__(self, algo, checkpoint_nr, metric_name, mean, std):
        self.algo = algo
        self.checkpoint_nr = checkpoint_nr
        self.metric_name = metric_name
        self.mean = mean
        self.std = std
