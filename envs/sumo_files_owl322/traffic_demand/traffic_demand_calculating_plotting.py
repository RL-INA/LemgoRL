import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PLOT = False
ROUTING = True
PEDESTRIANS = False

pd.options.display.width = 0
pd.set_option('display.max_rows', None)
idx = pd.IndexSlice

def routing_sorted_after_vehicles():
    # rename columns that they fit to taz-regions
    street_taz_mapping = {
        'Von RiWa-Str. nach Entruper Weg (Süd)': 'from_3_to_4',
        'Von RiWa-Str. nach Gosebrede': 'from_3_to_1',
        'Von RiWa-Str. nach Entruper Weg (Nord)': 'from_3_to_2',
        'Von Entruper Weg (Süd) nach Gosebrede': 'from_4_to_1',
        'Von Entruper Weg (Süd) nach Entruper Weg (Nord)': 'from_4_to_2',
        'Von Entruper Weg (Süd) nach RiWa-Str.': 'from_4_to_3',
        'Von Gosebrede nach Entruper Weg (Nord)': 'from_1_to_2',
        'Von Gosebrede nach RiWa-Str.': 'from_1_to_3',
        'Von Gosebrede nach Entruper Weg (Süd)': 'from_1_to_4',
        'Von Entruper Weg (Nord) nach RiWa-Str.': 'from_2_to_3',
        'Von Entruper Weg (Nord) nach Entruper Weg (Süd)': 'from_2_to_4',
        'Von Entruper Weg (Nord) nach Gosebrede': 'from_2_to_1',
    }
    df.rename(mapper=street_taz_mapping, axis=1, inplace=True)

    # rename vehicle names that they fit to sumo vehicle types
    veh_class_mapping = {
        'PKW + Transporter': 'car',
        'Fahrrad (auf Straße)': 'bicycle',
        'LKW': 'truck',
        'Lastzug': 'trailer',
        'Bus': 'bus',
        'Motorisierte Zweiräder': 'motorcycle',
        'Fußgänger + Fahrrad (auf Überweg)': 'pedestrian',
    }
    df.rename(mapper=veh_class_mapping, axis=0, inplace=True, level=1)

    # write od_flows.rou.xml
    veh_types = ['car', 'truck', 'trailer', 'bicycle', 'motorcycle']
    flow_id = 0
    PROB_NORM = 5*60 # since we have 5min time slots, the probability that a veh spaws = count / (5*60)

    flow_path = "od_flows.rou.xml"
    with open(flow_path, 'w') as file:
        file.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                   'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        for veh in veh_types:
            for _, row in df.loc[idx[:, veh], :].drop(columns="Total").iterrows():
                begin = row.name[0].split(' - ')[0]
                end = row.name[0].split(' - ')[1]
                for ind in row.index:
                    # probability = 0% are not accepted by sumo
                    if row[ind] > 0:
                        from_taz = ind.split('_')[1]
                        to_taz = ind.split('_')[3]
                        line = f'\t<flow id="{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" ' \
                               f'probability="{row[ind]/PROB_NORM}" ' \
                               f'type="{veh}" fromTaz="{from_taz}" toTaz="{to_taz}" departLane="free" departSpeed="max"/>\n'
                        file.write(line)
                        flow_id += 1
        file.write('</routes>')

def routing_sorted_after_times():
    '''
    Please use this version, since it sorts the flow-entries according to the departure time.
    :return:
    '''
    # rename columns that they fit to taz-regions
    street_taz_mapping = {
        'Von RiWa-Str. nach Entruper Weg (Süd)': 'from_3_to_4',
        'Von RiWa-Str. nach Gosebrede': 'from_3_to_1',
        'Von RiWa-Str. nach Entruper Weg (Nord)': 'from_3_to_2',
        'Von Entruper Weg (Süd) nach Gosebrede': 'from_4_to_1',
        'Von Entruper Weg (Süd) nach Entruper Weg (Nord)': 'from_4_to_2',
        'Von Entruper Weg (Süd) nach RiWa-Str.': 'from_4_to_3',
        'Von Gosebrede nach Entruper Weg (Nord)': 'from_1_to_2',
        'Von Gosebrede nach RiWa-Str.': 'from_1_to_3',
        'Von Gosebrede nach Entruper Weg (Süd)': 'from_1_to_4',
        'Von Entruper Weg (Nord) nach RiWa-Str.': 'from_2_to_3',
        'Von Entruper Weg (Nord) nach Entruper Weg (Süd)': 'from_2_to_4',
        'Von Entruper Weg (Nord) nach Gosebrede': 'from_2_to_1',
    }
    df.rename(mapper=street_taz_mapping, axis=1, inplace=True)

    # rename vehicle names that they fit to sumo vehicle types
    veh_class_mapping = {
        'PKW + Transporter': 'car',
        'Fahrrad (auf Straße)': 'bicycle',
        'LKW': 'truck',
        'Lastzug': 'trailer',
        'Bus': 'bus',
        'Motorisierte Zweiräder': 'motorcycle',
        'Fußgänger + Fahrrad (auf Überweg)': 'pedestrian',
    }
    df.rename(mapper=veh_class_mapping, axis=0, inplace=True, level=1)

    # write od_flows.rou.xml
    veh_types = ['car', 'truck', 'trailer', 'bicycle', 'motorcycle']
    times = []
    for t in df.index:
        times.append(t[0])
    times = sorted(list(set(times)))
    flow_id = 0
    PROB_NORM = 5 * 60  # since we have 5min time slots, the probability that a veh spaws = count / (5*60)
    REDUCTION_FACTOR = 0.90 # Reduce Traffic by this factor

    flow_path = "../od_flows.rou.xml"
    with open(flow_path, 'w') as file:
        file.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                   'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        for time in times:
            for _, row in df.loc[idx[time, veh_types], :].drop(columns="Total").iterrows():
                begin = row.name[0].split(' - ')[0]
                end = row.name[0].split(' - ')[1]
                for ind in row.index:
                    # probability = 0% are not accepted by sumo
                    if row[ind] > 0:
                        from_taz = ind.split('_')[1]
                        to_taz = ind.split('_')[3]
                        line = f'\t<flow id="{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" ' \
                               f'probability="{row[ind] / PROB_NORM * REDUCTION_FACTOR}" ' \
                               f'type="{row.name[1]}" fromTaz="{from_taz}" ' \
                               f'toTaz="{to_taz}" departLane="free" departSpeed="max"/>\n'
                        file.write(line)
                        flow_id += 1
        file.write('</routes>')

def pedestrians():
    global df
    street_taz_mapping = {
        'Von RiWa-Str. nach Entruper Weg (Süd)': 'from_3_to_4',
        'Von RiWa-Str. nach Gosebrede': 'from_3_to_1',
        'Von RiWa-Str. nach Entruper Weg (Nord)': 'from_3_to_2',
        'Von Entruper Weg (Süd) nach Gosebrede': 'from_4_to_1',
        'Von Entruper Weg (Süd) nach Entruper Weg (Nord)': 'from_4_to_2',
        'Von Entruper Weg (Süd) nach RiWa-Str.': 'from_4_to_3',
        'Von Gosebrede nach Entruper Weg (Nord)': 'from_1_to_2',
        'Von Gosebrede nach RiWa-Str.': 'from_1_to_3',
        'Von Gosebrede nach Entruper Weg (Süd)': 'from_1_to_4',
        'Von Entruper Weg (Nord) nach RiWa-Str.': 'from_2_to_3',
        'Von Entruper Weg (Nord) nach Entruper Weg (Süd)': 'from_2_to_4',
        'Von Entruper Weg (Nord) nach Gosebrede': 'from_2_to_1',
    }
    df.rename(mapper=street_taz_mapping, axis=1, inplace=True)

    # rename vehicle names that they fit to sumo vehicle types
    veh_class_mapping = {
        'PKW + Transporter': 'car',
        'Fahrrad (auf Straße)': 'bicycle',
        'LKW': 'truck',
        'Lastzug': 'trailer',
        'Bus': 'bus',
        'Motorisierte Zweiräder': 'motorcycle',
        'Fußgänger + Fahrrad (auf Überweg)': 'pedestrian',
    }
    df.rename(mapper=veh_class_mapping, axis=0, inplace=True, level=1)

    # select only pedestrian-relevant data
    crossings = ['from_3_to_1', 'from_4_to_2', 'from_1_to_3', 'from_2_to_4']
    df = df.loc[idx[:, 'pedestrian'], crossings]
    # print(df)

    # creating pedestrian_flow.rou.xml
    times = []
    for t in df.index:
        times.append(t[0])
    times = sorted(list(set(times)))

    flow_id = 0
    PROB_NORM = 5 * 60  # since we have 5min time slots, the probability that a veh spaws = count / (5*60)

    flow_path = "../pedestrian_flows.rou.xml"
    with open(flow_path, 'w') as file:
        file.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                   'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        for time in times:
            begin = time.split(' - ')[0]
            end = time.split(' - ')[1]
            line = ''
            line += f'\t<personFlow id="nsl_{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" probability="{df.loc[idx[time, "pedestrian"], "from_2_to_4"]/PROB_NORM/2}"><walk from="Gosebrede.S.1" to="Gosebrede.N.8"/></personFlow>\n' if df.loc[idx[time, "pedestrian"], "from_2_to_4"] > 0 else ''
            line += f'\t<personFlow id="snl_{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" probability="{df.loc[idx[time, "pedestrian"], "from_4_to_2"]/PROB_NORM/2}"><walk from="Gosebrede.N.8" to="Gosebrede.S.1"/></personFlow>\n' if df.loc[idx[time, "pedestrian"], "from_4_to_2"] > 0 else ''

            line += f'\t<personFlow id="wet_{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" probability="{df.loc[idx[time, "pedestrian"], "from_1_to_3"]/PROB_NORM /2}"><walk from="EntruperWeg.S.8" to="EntruperWeg.N.4"/></personFlow>\n' if df.loc[idx[time, "pedestrian"], "from_1_to_3"] > 0 else ''
            line += f'\t<personFlow id="ewt_{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" probability="{df.loc[idx[time, "pedestrian"], "from_3_to_1"]/PROB_NORM /2}"><walk from="EntruperWeg.N.4" to="EntruperWeg.S.8"/></personFlow>\n' if df.loc[idx[time, "pedestrian"], "from_3_to_1"] > 0 else ''

            line += f'\t<personFlow id="nsr_{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" probability="{df.loc[idx[time, "pedestrian"], "from_2_to_4"]/PROB_NORM /2}"><walk from="Richard-Wagner-Strasse.W.6" to="Richard-Wagner-Strasse.E.1"/></personFlow>\n' if df.loc[idx[time, "pedestrian"], "from_2_to_4"] > 0 else ''
            line += f'\t<personFlow id="snr_{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" probability="{df.loc[idx[time, "pedestrian"], "from_4_to_2"]/PROB_NORM /2}"><walk from="Richard-Wagner-Strasse.E.1" to="Richard-Wagner-Strasse.W.6"/></personFlow>\n' if df.loc[idx[time, "pedestrian"], "from_4_to_2"] > 0 else ''

            line += f'\t<personFlow id="web_{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" probability="{df.loc[idx[time, "pedestrian"], "from_1_to_3"]/PROB_NORM /2}"><walk from="EntruperWeg.S.9" to="EntruperWeg.N.3"/></personFlow>\n' if df.loc[idx[time, "pedestrian"], "from_1_to_3"] > 0 else ''
            line += f'\t<personFlow id="ewb_{flow_id}" begin="{begin + ":00"}" end="{end + ":00"}" probability="{df.loc[idx[time, "pedestrian"], "from_3_to_1"]/PROB_NORM /2}"><walk from="EntruperWeg.N.3" to="EntruperWeg.S.9"/></personFlow>\n' if df.loc[idx[time, "pedestrian"], "from_3_to_1"] > 0 else ''
            file.write(line)
            flow_id += 1
        file.write('</routes>')


def bar_plot():
    times = []
    for t in df.index:
        times.append(t[0])
    times = sorted(list(set(times)))

    veh_list = ['PKW + Transporter', 'LKW', 'Lastzug', 'Bus', 'Motorisierte Zweiräder', 'Fahrrad (auf Straße)',
                'Fußgänger + Fahrrad (auf Überweg)']
    for t in times:
        df.loc[idx[t, 'Total_correct'], "Total"] = np.sum([df.loc[idx[t, veh], "Total"] for veh in veh_list])

    veh_list.remove('Bus')
    for t in times:
        df.loc[idx[t, 'Total_correct_without_busses'], "Total"] = np.sum(
            [df.loc[idx[t, veh], "Total"] for veh in veh_list])

    # STACKED HISTOGRAM
    times_hist = [t.split('-') for t in times]
    times_bins = [t[0] for t in times_hist] + [times_hist[-1][-1]]

    # PIVOTTING
    vals_columns = ['PKW + Transporter', 'LKW', 'Lastzug', 'Motorisierte Zweiräder', 'Fahrrad (auf Straße)', 'Fußgänger + Fahrrad (auf Überweg)']
    df_bars = df['Total'].unstack()[vals_columns]
    df_bars['begin_times'] = times_bins[:-1]
    df_bars.set_index(keys='begin_times', inplace=True)
    veh_class_mapping = {
        'PKW + Transporter': 'car',
        'Fahrrad (auf Straße)': 'bicycle',
        'LKW': 'truck',
        'Lastzug': 'truck with trailer',
        'Bus': 'bus',
        'Motorisierte Zweiräder': 'motorcycle',
        'Fußgänger + Fahrrad (auf Überweg)': 'pedestrian',
    }
    df_bars.rename(mapper=veh_class_mapping, axis=1, inplace=True)
    print(df_bars)

    # STACK BAR PLOT
    # fig, ax = plt.subplots()
    ax = df_bars.plot(kind='bar', stacked=True,
                      xlabel='time', ylabel='amount of road users',
                      position=-0.1, width=0.8)
    ax.legend(df_bars.columns)
    pad_inches=0.1
    plt.savefig('traffic_volume_owl322.pdf', bbox_inches='tight', pad_inches=pad_inches)
    # plt.savefig('traffic_volume_owl322.png', bbox_inches='tight', pad_inches=pad_inches)
    plt.close()

if __name__ == '__main__':
    path = "03_OWL_322_ohne_LZ.xlsx"
    df = pd.read_excel(path, sheet_name=0, header=0, index_col=[0, 1], engine="xlrd")
    df.fillna(value=0, axis=0, inplace=True)
    df = df.astype('int')

    if (PLOT and ROUTING) or (PLOT and PEDESTRIANS) or (PEDESTRIANS and ROUTING):
        raise Exception('Please use only option from PLOT, ROUTING or PEDESTRIANS at the same time!')
    if PLOT: bar_plot()
    if ROUTING: routing_sorted_after_times()
    if PEDESTRIANS: pedestrians()