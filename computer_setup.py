import subprocess
import time
import os, socket, sys
import re
import signal
from pathlib import PosixPath


def init_pc(
    virtual=False,
    SLS_RECEIVER_PORT="1954",
    PROCESSING_RX_IP="192.168.2.200",
    PROCESSING_RX_PORT="50003",
    PROCESSING_TX_IP="192.168.1.118",
    PROCESSING_TX_PORT="50001",
    PROCESSING_CORES="20",
    CONFIG_PATH_REAL="/home/moench/detector/moench_2021.config",
    CONFIG_PATH_VIRTUAL="/home/moench/detector/moench_2021_virtual.config",
    EXECUTABLES_PATH="/opt/slsDetectorPackage/build/bin/",
    ROOT_USERNAME="dummy_user",
    ROOT_PASSWORD="dummy_password",
    CLEAR_STARTUP=False,
):
    start_10g_interface(ROOT_USERNAME, ROOT_PASSWORD)
    if CLEAR_STARTUP:
        kill_all_pc_processes(ROOT_USERNAME, ROOT_PASSWORD)
    if virtual:
        CONFIG_PATH = CONFIG_PATH_VIRTUAL  # for virtual detector
        subprocess.Popen(
            f"{EXECUTABLES_PATH}moenchDetectorServer_virtual",
            shell=False,
        )
        time.sleep(5)
        print("configured for virtual detector")

    else:
        CONFIG_PATH = CONFIG_PATH_REAL  # for real (hardware) detector
        print("configured for real detector")
    # CONFIG_PATH = "/home/moench/detector/moench_2021.config" #for real detector
    # configured for moench pc only
    subprocess.Popen(
        f'su - {ROOT_USERNAME} -c "sudo {EXECUTABLES_PATH}slsReceiver -t {SLS_RECEIVER_PORT}" <<< {ROOT_PASSWORD}',
        shell=True,
    )
    print("started slsReceicver")
    time.sleep(5)
    subprocess.Popen(
        [
            f"{EXECUTABLES_PATH}moench03ZmqProcess",
            PROCESSING_RX_IP,
            PROCESSING_RX_PORT,
            PROCESSING_TX_IP,
            PROCESSING_TX_PORT,
            PROCESSING_CORES,
        ],
    )
    subprocess.call([f"{EXECUTABLES_PATH}sls_detector_put", "config", CONFIG_PATH])
    time.sleep(5)
    print("Both processses are running")
    if virtual:
        subprocess.call([f"{EXECUTABLES_PATH}sls_detector_put", "config", CONFIG_PATH])
        print("Uploaded the config the 2nd time for virtual")
    print("Both processses are running")
    return is_pc_ready()


def kill_all_pc_processes(ROOT_USERNAME, ROOT_PASSWORD):
    kill_processes_by_name(
        "slsReceiver",
        root_username=ROOT_USERNAME,
        root_password=ROOT_PASSWORD,
        sudo=True,
    )
    kill_processes_by_name("moench03ZmqProcess")
    kill_processes_by_name("moenchDetectorServer_virtual")


def deactivate_pc(ROOT_USERNAME, ROOT_PASSWORD):
    kill_all_pc_processes(ROOT_USERNAME, ROOT_PASSWORD)


def is_sls_running():
    return is_process_running("slsReceiver")


def is_zmq_running():
    return is_process_running("moench04ZmqProcess")


def is_pc_ready():
    if is_sls_running() and is_zmq_running():
        return True
    else:
        return False


def is_process_running(name):
    try:
        lines = os.popen("pgrep -f %s" % name)
        if not list(lines):
            return False
        else:
            return True
    except:
        print("Error occurred while process running check")


def kill_processes_by_name(
    name, sudo=False, root_username="root", root_password="pass"
):
    try:
        for line in os.popen("pgrep -f %s" % name):
            pid = int(line)
            if sudo:
                subprocess.call(
                    f'su - {root_username} -c "sudo kill -9 {pid}" <<< {root_password}',
                    shell=True,
                )
            else:
                os.kill(pid, signal.SIGKILL)
    except:
        print("Error occurred while killing process")


def start_10g_interface(root_username="root", root_password="pass"):
    subprocess.call(
        f'su - {root_username} -c "sudo ifup em2" <<< {root_password}',
        shell=True,
    )
