"""
Models for SUMO and Lisa+ Virtual Controller information mapping
@author: Vishal Rangras (Fraunhofer IOSB-INA in Lemgo)
@email: vishal.rangras@iosb-ina.fraunhofer.de
"""
from deprecated import deprecated


class ControllerUnit:

    def __init__(self, z_nr, f_nr, sumo_tl_id):
        self.z_nr = z_nr
        self.f_nr = f_nr
        self.sumo_tl_id = sumo_tl_id
        self.taskID = -1
        self.lisa_sumo_mapping = None
        self.lisa_sgr_seq = None
        self.sgr_link_indices_dict = None


@deprecated
class SumoOmtcDetector:

    def __init__(self, obj_id, lisa_id, sumo_id, controlled_node):
        pass


@deprecated
class SumoOmtcVariable:

    def __init__(self, varID, name, varType, controlled_node):
        pass

@deprecated
class SumoOmtcSignalGroup:

    def __init__(self, lisa_id, sumo_id, controlled_node):
        pass