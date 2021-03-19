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