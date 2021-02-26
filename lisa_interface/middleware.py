"""
Middleware to facilitate communication between SUMO and LISA+ Virtual Controller
@author: Vishal Rangras (Fraunhofer IOSB-INA in Lemgo)
@email: vishal.rangras@iosb-ina.fraunhofer.de
"""

import requests
import re
import subprocess
import time
import psutil
import logging
from pathlib import Path
import xml.dom.minidom
from lisa_interface.model import ControllerUnit, SumoOmtcDetector, SumoOmtcVariable, SumoOmtcSignalGroup
from deprecated import deprecated


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-middleware-'+timestamp+'.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()


class LisaInterfaceManager:

    Messages = {
        "MSG_GetTaskListRequest": "<GetTaskListRequest xmlns=\"http://www.schlothauer.de/omtc/services\">"
                                  "<Detail>false</Detail><StgKennung><ZNr>%d</ZNr><FNr>%d</FNr></StgKennung>"
                                  "</GetTaskListRequest>",
        "MSG_SetDataDirRequest": "<SetDataDirRequest xmlns=\"http://www.schlothauer.de/omtc/services\">"
                                 "<Value>%s</Value></SetDataDirRequest>",
        "MSG_RemoveTaskRequest": "<RemoveTaskRequest xmlns=\"http://www.schlothauer.de/omtc/services\">"
                                 "<ID>%s</ID></RemoveTaskRequest>",
        "MSG_SetTaskRequest": (
            """<SetTaskRequest xmlns=\"http://www.schlothauer.de/omtc/services\">
                <ID>%d</ID>
                <StgKennung><ZNr>%d</ZNr><FNr>%d</FNr></StgKennung>
                <Callback><URL>%s</URL></Callback>
                <Cycle><IntervallSec>%d</IntervallSec></Cycle>"
                %s
               </SetTaskRequest>
            """
        ),
        "MSG_TaskElement": "<TaskElement><MessageType>%s</MessageType></TaskElement>",
        "MSG_GetObjectListRequest": "<ObjectListRequest xmlns=\"http://www.schlothauer.de/omtc/services\">"
                                    "<StgKennung><ZNr>%d</ZNr><FNr>%d</FNr></StgKennung></ObjectListRequest>",
        "MSG_Message": "<Message xmlns=\"http://www.schlothauer.de/omtc/services\"><Msg>%s</Msg></Message>"
    }

    MessageToService = {
        "MSG_GetTaskListRequest": "/services/PDService/getTaskList",
        "MSG_SetDataDirRequest": "/services/DDService/setDataDir",
        "MSG_RemoveTaskRequest": "/services/PDService/removeTask",
        "MSG_SetTaskRequest": "/services/PDService/setTask",
        "MSG_GetObjectListRequest": "/services/PDService/getObjectList",
        "MSG_Message": "/services/PDCallback/putMessage"
    }

    ParameterAttributes = {
        "detectorMapping": "LISA+_detectorMapping",
        "signalGroupMapping": "LISA+_signalGroupMapping",
        "signalGroupOrder": "LISA+_signalGroupOrder",
        "signalPriority": "LISA+_signalPriority",
        "variableMapping": "LISA+_variableMapping",
        "fileDir": "LISA+_fileDir",
        "externalID": "LISA+_externalID",
        "host": "LISA+_host",
        "port": "LISA+_port",
        "controlMode": "LISA+_controlMode",
        "coordinated": "LISA+_coordinated",
        "sp": "LISA+_sp",
        "va": "LISA+_va",
        "oev": "LISA+_oev",
        "iv": "LISA+_iv"
    }

    """
        SUMO: G green, g permissive green, y yellow, r red, u red/yellow, o off, O off with right of way
        see https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions
        
        r -	'red light' for a signal - vehicles must stop
        y -	'amber (yellow) light' for a signal - vehicles will start to decelerate if far away from the junction,
            otherwise they pass
        g -	'green light' for a signal, no priority - vehicles may pass the junction if no vehicle uses
            a higher priorised foe stream, otherwise they decelerate for letting it pass.
            They always decelerate on approach until they are within the configured visibility distance
        G -	'green light' for a signal, priority - vehicles may pass the junction
        s -	'green right-turn arrow' requires stopping - vehicles may pass the junction if no vehicle uses
            a higher priorised foe stream. They always stop before passing.
            This is only generated for junction type traffic_light_right_on_red.
        u -	'red+yellow light' for a signal, may be used to indicate upcoming green phase but vehicles may
            not drive yet (shown as orange in the gui)
        o -	'off - blinking' signal is switched off, blinking light indicates vehicles have to yield
        O -	'off - no signal' signal is switched off, vehicles have the right of way
    """
    LisaToSumoSignals = {
        3: "r",  # OCIT-Farbbild Rot
        15: "u",  # OCIT-Farbbild Rot/Gelb
        12: "y",  # OCIT-Farbbild Gelb
        48: "g",  # OCIT-Farbbild Gruen
        0: "O",  # OCIT-Farbbild Dunkel
        51: "o",  # OCIT-Farbbild Rot/Gruen # not for usecase
        60: "o",  # OCIT-Farbbild Gelb/Gruen # only in austria
        2: "r",  # OCIT-Farbbild Rot-Blinken
        8: "o",  # OCIT-Farbbild Gelb-Blinken
        32: "o",  # OCIT-Farbbild Gruen-Blinken
        # 8: "g",  # OCIT-Farbbild Gelb-Blinken #von M. Barthauer
        # 32: "g",  # OCIT-Farbbild Gruen-Blinken #von M. Barthauer
    }

    DESIRED_PH_TO_LISA_PH_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}

    LisaControlFilePattern = re.compile(r"z([0-9]+)_fg([0-9]+)\.xml")
    LisaControllerPattern = re.compile(r"z([0-9]+)_fg([0-9]+)")
    PutMessageResponsePattern = re.compile(r"^([0-9]+):(\{.+\})(\{.+\})(\{.+\})(\{.*\})(\{.*\})(\{.*\})$",
                                           flags=re.DOTALL)

    controlOptions = {"controlMode": 5, "sp": 11, "va": 2, "iv": 2, "oev": 2, "coordinated": 2}
    detString = ""
    msgTypeInit = 'Init'
    msgTypeRun = 'Run'

    def __init__(self, host, port, server_path, data_dir, controlled_nodes, lisa_cfg):
        self.host = host
        self.port = port
        self.server_path = server_path
        self.oml_fg_server = None
        self.data_dir = data_dir
        self.controlled_nodes = controlled_nodes
        self.controller_unit_dict = {}
        self.detector_unit_list = []
        self.phases = []
        self.signalPlans = []
        self.signalGroups = {}
        self.variables = {}
        self.variableOrder = []
        self.signalGroups = {}
        self.lisa_sumo_sgr_dict = {}
        self.lisa_cfg = lisa_cfg
        self.pedestrian_demand = ['F', 'F', 'F', 'F']

    def __start_oml_fg_server(self, invisible=True):
        java_exec = "javaw.exe" if invisible else "java.exe"
        java_args = [java_exec]
        java_args.extend(["-jar", "-Xmx1024m", "-Xms512m", self.server_path])

        # check for already running process which might block the port?
        for proc in psutil.process_iter():
            # Check if process name contains the given name string.
            if java_exec in proc.name().lower():
                proc.kill()
        try:
            self.oml_fg_server = subprocess.Popen(java_args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            time.sleep(5)
        except PermissionError as pe:
            logging.error("An exception occurred. {0}".format(pe))
            exit(-1)

    def initialize_lisa_context(self, phase_dict):

        # 0. Start OmlFgServer
        self.__start_oml_fg_server()

        # 1. Set Data Dir
        data_dir_body = self.Messages["MSG_SetDataDirRequest"] % self.data_dir
        data_dir_response = self.__invoke_service(self.MessageToService["MSG_SetDataDirRequest"], data_dir_body)
        logger.debug('<SetDataDir> Response received : %s' % data_dir_response)

        for controlled_node in self.controlled_nodes:
            lisa_id = self.lisa_cfg[controlled_node]['id']['lisa_id']
            match = re.search(self.LisaControllerPattern, lisa_id)
            if match:
                z_nr, f_nr = match.group(1, 2)
                controller_unit = ControllerUnit(z_nr, f_nr, controlled_node)
            else:
                raise Exception(f"Lisa+ controller ID could not be found for controlled node: {controlled_node}."
                                f" Please configure it in lisa_config.yaml.")

            # 2. Get Task List
            task_list_body = self.Messages["MSG_GetTaskListRequest"] %\
                (int(controller_unit.z_nr), int(controller_unit.f_nr))
            task_list_response = self.__invoke_service(self.MessageToService["MSG_GetTaskListRequest"], task_list_body)
            logger.debug('<GetTaskList> Response received : %s' % task_list_response)
            task_id_list = []
            dom_tree = xml.dom.minidom.parseString(task_list_response)
            task_ids = dom_tree.getElementsByTagName("ns2:ID")
            for task_id in task_ids:
                task_id_list.append(task_id.childNodes[0].nodeValue)

            # 3. Remove Tasks if the list is not empty
            # removeTask (remove existing tasks before starting a new one)
            for task_id in task_ids:
                remove_task_body = self.Messages["MSG_RemoveTaskRequest"] % task_id
                remove_task_response = self.__invoke_service(self.MessageToService["MSG_RemoveTaskRequest"],
                                                             remove_task_body)
                logger.debug('<RemoveTask> Response received : %s' % remove_task_response)

            # 4. Set Tasks
            message_types = ["MeldungType", "WunschVektorType", "DetFlType", "OevTelegrammType", "APWertZustType",
                             "IstvektorProjType"]
            task_element_list = [self.Messages["MSG_TaskElement"] % messageType for messageType in message_types]
            task_elements = "".join(task_element_list)
            set_tasks_body = self.Messages["MSG_SetTaskRequest"] % \
                (0, int(controller_unit.z_nr), int(controller_unit.f_nr),
                    "%s:%d" % (self.host, 9010), 60, task_elements)

            set_tasks_response = self.__invoke_service(self.MessageToService["MSG_SetTaskRequest"], set_tasks_body)
            logger.debug('<SetTasks> Response received : %s' % set_tasks_response)
            try:
                dom_tree = xml.dom.minidom.parseString(set_tasks_response)
                task_nodes = dom_tree.getElementsByTagName("ns2:ID")
                for task_node in task_nodes:
                    controller_unit.taskID = int(task_node.firstChild.nodeValue)
                    break
            except Exception as e:
                logger.error(e)

            # 5. Get Object List
            # get_object_list_body = self.Messages["MSG_GetObjectListRequest"]
            # % (int(controllerUnit.z_nr), int(controllerUnit.f_nr))
            # get_object_list_response = self.invoke_service(self.MessageToService["MSG_GetObjectListRequest"]
            # , get_object_list_body)
            # logger.debug('<Get Object List> Response received : %s' % get_object_list_response)
            # self.process_object_list_req_sgr(controlled_node, get_object_list_response)

            # 6. put Message
            controller_unit.lisa_sumo_mapping = self.lisa_cfg[controlled_node]['signal']['lisa_sumo_mapping']
            controller_unit.lisa_sgr_seq = self.lisa_cfg[controlled_node]['signal']['lisa_sgr_seq'].split(",")
            self.__init_lisa_sumo_sgr_mapping(controller_unit)

            sim_time = 0
            ap_val = self.__calc_ap_vector(phase_dict[controlled_node], self.pedestrian_demand, sim_time)
            put_message_body = self.__prepare_put_message_body(controller_unit, sim_time, self.msgTypeInit, ap_val)
            put_message_response = self.__invoke_service(self.MessageToService["MSG_Message"], put_message_body)
            logger.debug('<Put Message> Response received : %s' % put_message_response)
            self.__process_put_message_response(put_message_response, controller_unit)
            self.controller_unit_dict[controlled_node] = controller_unit

    def get_sgr_states(self, controlled_node, phase, pedestrian_demand, sim_time):
        if pedestrian_demand is None:
            pedestrian_demand = self.pedestrian_demand
        controller_unit = self.controller_unit_dict[controlled_node]
        ap_val = self.__calc_ap_vector(phase, pedestrian_demand, sim_time)
        put_message_body = self.__prepare_put_message_body(controller_unit, sim_time, self.msgTypeRun, ap_val)
        put_message_response = self.__invoke_service(self.MessageToService["MSG_Message"], put_message_body)
        logger.debug('<Put Message> Response received : %s' % put_message_response)
        return self.__process_put_message_response(put_message_response, controller_unit)

    def __calc_ap_vector(self, phase, pedestrian_demand, sim_sec):
        logger.debug(f"The desired phase is: {phase}")
        logger.debug(f"The computed lisa+ phase wish is :{self.DESIRED_PH_TO_LISA_PH_MAP[phase]}")
        logger.debug(f"The pedestrian demand vector is :{pedestrian_demand}")
        ap_vector = f'{sim_sec}/{self.DESIRED_PH_TO_LISA_PH_MAP[phase]}/{pedestrian_demand[0]}' \
                    f'/{pedestrian_demand[1]}/{pedestrian_demand[2]}/{pedestrian_demand[3]}'
        logger.debug(f"The computed ap_vector is :{ap_vector}")

        return ap_vector

    def __prepare_put_message_body(self, controller_unit, sim_time, msg_type, ap_val):
        msg = "%d %d %d %d:%d{\"%s\"}{%d;%d;%d;%d;%d;%d;%d;%d}{%s}{}{%s}{}" % \
              (controller_unit.taskID, sim_time, 1, 0, sim_time, msg_type, self.controlOptions["controlMode"], 1,
               self.controlOptions["sp"], 1, self.controlOptions["va"], self.controlOptions["iv"],
               self.controlOptions["oev"], self.controlOptions["coordinated"], self.detString, ap_val)
        logger.debug(f"Lisa+ putMessage Request: {msg}")
        return self.Messages["MSG_Message"] % msg

    def __process_put_message_response(self, put_message_response, controller_unit):
        dom_tree = xml.dom.minidom.parseString(put_message_response)
        internal_cmd = dom_tree.documentElement.firstChild.nodeValue
        logger.debug(f"Lisa+ putMessage Response: {internal_cmd}")

        curly_braces = re.search(self.PutMessageResponsePattern, internal_cmd)
        if curly_braces:
            try:
                sec_string, cycle_sec_string, flag_string, signal_states_string, \
                    output_string, phases_string, ap_string = curly_braces.group(1, 2, 3, 4, 5, 6, 7)

                signal_states_string = signal_states_string[1:-1]
                sumo_signals_str = self.__map_lisa_signals_to_sumo(signal_states_string, controller_unit)
                phases_string = phases_string[1:-1]
                output_string = output_string[1:-1]
                ap_string = ap_string[1:-1]
                logger.debug(f"Lisa Signal States: {signal_states_string}")
                logger.debug(f"Sumo Signal States: {sumo_signals_str}")
                logger.debug(f"Phase : {phases_string}")
                logger.debug(f"Output String: {output_string}")
                logger.debug(f"AP Values : {ap_string}")
                return sumo_signals_str, phases_string, output_string, ap_string
            except IndexError as ie:
                raise ie

    @staticmethod
    def __init_lisa_sumo_sgr_mapping(controller_unit):
        lisa_sumo_mapping = controller_unit.lisa_sumo_mapping
        sgr_link_indices_dict = {}
        for sgr, links in lisa_sumo_mapping.items():
            logger.debug(f"Processing the sgr: {sgr.strip()}")
            sgr_link_indices_dict[sgr.strip()] = [int(x.strip()) for x in str(links).split(",")]
            logger.debug(f"Link Indices for Sgr {sgr.strip()} are: {sgr_link_indices_dict[sgr.strip()]}")
        controller_unit.sgr_link_indices_dict = sgr_link_indices_dict

    def __map_lisa_signals_to_sumo(self, signal_states_string, controller_unit):
        signals = [int(i) for i in signal_states_string.split("/")]
        sgr_sumo_state_list = [self.LisaToSumoSignals[x] for x in signals]
        sgr_link_indices_dict = controller_unit.sgr_link_indices_dict
        lisa_sgr_seq = controller_unit.lisa_sgr_seq

        link_state_dict = {}
        if len(sgr_sumo_state_list) != len(sgr_link_indices_dict.keys()):
            raise Exception("The lisa_sgr_seq property does not match the signal groups defined in Lisa+ program. "
                            "Please configure lisa_sgr_seq correctly in lisa_config.yaml!")

        for num, sgr_sumo_state in enumerate(sgr_sumo_state_list):
            link_indices = sgr_link_indices_dict[lisa_sgr_seq[num].strip()]
            if -1 in link_indices:
                continue
            if lisa_sgr_seq[num].strip() == 'KL03':
                if sgr_sumo_state.lower() == 'g' and sgr_sumo_state_list[num-1] == 'g':
                    kl03_sumo_state = 'G'
                else:
                    kl03_sumo_state = sgr_sumo_state_list[num-1]
                for link in link_indices:
                    link_state_dict[link] = kl03_sumo_state
            else:
                for link in link_indices:
                    link_state_dict[link] = sgr_sumo_state

        state_list = []
        for link, state in sorted(link_state_dict.items()):
            state_list.append(state)

        return "".join(state_list)

    @deprecated
    def __convert_lisa_signals_to_sumo(self, singnal_states_string, controller_unit):
        signals = [int(i) for i in singnal_states_string.split("/")]
        sumo_signals = [self.LisaToSumoSignals[x] for x in signals]
        lisa_sumo_association = controller_unit.lisa_sumo_association
        sumo_links_state = []
        index = 0
        for repeat_value in lisa_sumo_association:
            if repeat_value == 0:
                continue
            for i in range(repeat_value):
                sumo_links_state.append(sumo_signals[index])
            index += 1
        return "".join(sumo_links_state)

    @deprecated
    def __process_object_list_req_sgr(self, controlled_node, get_object_list_result):
        try:
            dom_tree = xml.dom.minidom.parseString(get_object_list_result)
        except Exception as e:
            raise e

        obj_nodes = dom_tree.getElementsByTagName("ns2:Objs")
        lisa_pos = 0

        for obj_node in obj_nodes:
            obj_id = None
            node_type = None
            name = None
            for child_node in obj_node.childNodes:
                if child_node.tagName == "ns2:ID":
                    obj_id = int(child_node.firstChild.nodeValue)
                elif child_node.tagName in ["ns2:Sgr"]:
                    node_type = child_node.tagName
                    if len(child_node.childNodes) > 0:
                        sub_child_node = child_node.getElementsByTagName(child_node.tagName)[0]
                        name = str(sub_child_node.firstChild.nodeValue)
                    if obj_id is not None:
                        break

            if node_type == "ns2:Sgr":
                if name is not None and name in list(self.lisa_sumo_sgr_dict[controlled_node].keys()):
                    lisa_id = name
                    sumo_id = self.lisa_sumo_sgr_dict[controlled_node][lisa_id]
                    omtc_signal_group = SumoOmtcSignalGroup(lisa_id, sumo_id, controlled_node)
                    omtc_signal_group.lisaPos = lisa_pos
                    self.signalGroups[controlled_node+'_'+str(obj_id)] = omtc_signal_group
                    lisa_pos += 1

    @deprecated
    def __process_object_list_req(self, controlled_node, get_object_list_result):
        try:
            dom_tree = xml.dom.minidom.parseString(get_object_list_result)
        except Exception as e:
            raise e

        obj_nodes = dom_tree.getElementsByTagName("ns2:Objs")
        lisa_pos = 0

        lisa_det_seq = None
        sumo_det_seq = None
        try:
            lisa_det_seq = [x.strip() for x in self.lisa_cfg[controlled_node]['detector']['lisa_det_seq'].split(",")]
            sumo_det_seq = [x.strip() for x in self.lisa_cfg[controlled_node]['detector']['sumo_det_seq'].split(",")]
            if (len(lisa_det_seq)) != len(sumo_det_seq):
                raise Exception("Please configure same number of lisa and sumo detectors in lisa_config.yaml")
        except KeyError as e:
            logger.error("Detectors are not configured in lisa_config.yaml. "
                         "The program will continue without detectors", e)

        for objNode in obj_nodes:
            obj_id = None
            node_type = None
            name = None
            for childNode in objNode.childNodes:
                if childNode.tagName == "ns2:ID":
                    obj_id = int(childNode.firstChild.nodeValue)
                elif childNode.tagName in ["ns2:Det", "ns2:Pha", "ns2:Spl", "ns2:Sgr", "ns2:APWert"]:
                    node_type = childNode.tagName
                    if len(childNode.childNodes) > 0:
                        sub_child_node = childNode.getElementsByTagName(childNode.tagName)[0]
                        name = str(sub_child_node.firstChild.nodeValue)
                    if obj_id is not None:
                        break

            # Process Detectors
            if node_type == "ns2:Det" and lisa_det_seq is not None and sumo_det_seq is not None:
                if len(lisa_det_seq) != len(sumo_det_seq):
                    raise Exception("Please configure same number of lisa and sumo detectors in lisa_config.yaml")
                elif name is not None and name in lisa_det_seq:
                    lisa_id = name
                    sumo_id = sumo_det_seq[lisa_det_seq.index(lisa_id)]
                    sumo_omtc_detector = SumoOmtcDetector(obj_id, lisa_id, sumo_id, controlled_node)
                    self.detector_unit_list.append(sumo_omtc_detector)

            # Process Phases
            elif node_type == "ns2:Pha":
                self.phases.append(obj_id)

            # Process Signal Plans
            elif node_type == "ns2:Spl":
                self.signalPlans.append(obj_id)

            # Process AP Values
            elif node_type == "ns2:APWert":
                var_type = None
                param_nodes = objNode.getElementsByTagName("ns2:Para")
                for paramNode in param_nodes:
                    param_name = str(paramNode.getElementsByTagName("ns2:Name")[0].firstChild.nodeValue)
                    if param_name == "Typ":
                        var_type = str(paramNode.getElementsByTagName("ns2:StrValue")[0].firstChild.nodeValue)
                        break

                # put everything into a data structure
                if var_type is not None:
                    sumo_omtc_variable = SumoOmtcVariable(obj_id, name, var_type, controlled_node)
                    self.variables[controlled_node+'_'+name] = sumo_omtc_variable
                    self.variableOrder.append(name)

            elif node_type == "ns2:Sgr":
                if name is not None and name in list(self.lisa_sumo_sgr_dict[controlled_node].keys()):
                    lisa_id = name
                    sumo_id = self.lisa_sumo_sgr_dict[controlled_node][lisa_id]
                    omtc_signal_group = SumoOmtcSignalGroup(lisa_id, sumo_id, controlled_node)
                    omtc_signal_group.lisaPos = lisa_pos
                    self.signalGroups[controlled_node+'_'+str(obj_id)] = omtc_signal_group
                    lisa_pos += 1

    def __invoke_service(self, endpoint, body):
        response = ""
        server_addr = "http://%s:%s" % (self.host, self.port)
        connected = False
        try_count = 0
        while not connected:
            try:
                headers = {'Accept': '*/*', 'Content-Type': 'text/xml;charset=\"utf-8\"',
                           'Accept-Encoding': 'gzip/deflate', 'Host': "%s:%s" % (self.host, self.port),
                           'Content-Length': str(len(body))}
                request = requests.post("%s%s" % (server_addr, endpoint), data=body, headers=headers)
                response = request.text
                if "Error 404" in response:
                    raise Exception("Error 404")
                else:
                    connected = True
            except Exception as e:
                logger.error("An exception occurred. {0}".format(e))
                try_count += 1
                time.sleep(3)
        return response
