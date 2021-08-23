import subprocess
import psutil
import os
import platform
from pathlib import Path
import time
import logging


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-tsc_util-'+timestamp+'.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()

WORKER_PORT_DICT = {1: 59081, 2: 59082, 3: 59083, 4: 59084, 5: 59085, 6: 59086, 7: 59087, 8: 59088,
                    9: 59089, 10: 59090, 11: 59091, 12: 59092, 13: 59093, 14: 59094, 15: 59095, 16: 59096,
                    17: 59097, 18: 59098, 19: 59099, 20: 59100, 21: 59101, 22: 59102, 23: 59103, 24: 59104,
                    25: 59105, 26: 59106, 27: 59107, 28: 59108, 29: 59109, 30: 59110, 31: 59111, 32: 59112}

VC_JAR_NAME = 'OmlFgServer.jar'


def start_oml_fg_server(algo_dict, server_path, starting_port, invisible=True):
    port_counter = 1
    for worker_index, port in WORKER_PORT_DICT.items():

        if port < starting_port:
            continue

        if port_counter > 2 and algo_dict['num_workers'] == 0:
            break
        elif port_counter > algo_dict['num_workers'] != 0:
            break

        if platform.system() == 'Linux':
            java_exec = "java"
            java_args = ["nohup", java_exec]
        else:
            java_exec = "javaw.exe" if invisible else "java.exe"
            java_args = [java_exec]
        java_args.extend(["-jar", "-Xmx1024m", "-Xms512m", server_path+'/'+str(port)+'/'+VC_JAR_NAME])

        try:
            if platform.system() == 'Linux':
                oml_fg_server = subprocess.Popen(java_args, preexec_fn=os.setpgrp)
            else:
                oml_fg_server = subprocess.Popen(java_args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                print("PID is:"+str(oml_fg_server.pid))
            logger.info(f"OmlFgServer started successfully with pid: {oml_fg_server.pid},"
                        f" on port: {port}, for worker: {port_counter}")
            port_counter += 1
        except PermissionError as pe:
            logger.error("An exception occurred. {0}".format(pe))
            exit(-1)
    time.sleep(10)


def shutdown_oml_fg_server(invisible=True):
    if platform.system() == 'Linux':
        java_exec = "java"
    else:
        java_exec = "javaw.exe" if invisible else "java.exe"
    logger.info("Checking if process needs to be killed...")
    for proc in psutil.process_iter():
        # Check if process name contains the given name string.
        if java_exec in proc.name().lower():
            logger.info(f"Killing the process: {proc.pid}")
            proc.kill()
    time.sleep(10)
