#!/bin/python3
from numpy import tri
from tango import AttrWriteType, DevState, DispLevel, GreenMode, AttrDataFormat
from tango.server import Device, attribute, command, pipe, device_property
from slsdet import Moench, runStatus, timingMode, detectorSettings, frameDiscardPolicy
from _slsdet import IpAddr
import time
import os, sys
import re
import computer_setup
from pathlib import PosixPath
from enum import Enum, IntEnum
from bidict import bidict
import asyncio
import numpy as np
import random
import datetime
from skimage.io import imread


class MoenchDetectorControl(Device):
    _tiff_fullpath_last = ""
    _last_triggers = ""
    _last_image = np.zeros([400, 400], dtype=np.int)
    green_mode = GreenMode.Asyncio

    class FrameMode(IntEnum):
        # hence detectormode in slsdet uses strings (not enums) need to be converted to strings
        # RAW = "raw"
        # FRAME = "frame"
        # PEDESTAL = "pedestal"
        # NEWPEDESTAL = "newPedestal"
        # NO_FRAME_MODE = "noFrameMode"
        RAW = 0
        FRAME = 1
        PEDESTAL = 2
        NEWPEDESTAL = 3
        NO_FRAME_MODE = 4

    frameMode_bidict = bidict(
        {
            FrameMode.RAW: "raw",
            FrameMode.FRAME: "frame",
            FrameMode.PEDESTAL: "pedestal",
            FrameMode.NEWPEDESTAL: "newPedestal",
            FrameMode.NO_FRAME_MODE: "noFrameMode",
        }
    )

    class DetectorMode(IntEnum):
        # hence detectormode in slsdet uses strings (not enums) need to be converted to strings
        # COUNTING = "counting"
        # ANALOG = "analog"
        # INTERPOLATING = "interpolating"
        # NO_DETECTOR_MODE = "noDetectorMode"
        COUNTING = 0
        ANALOG = 1
        INTERPOLATING = 2
        NO_DETECTOR_MODE = 3

    detectorMode_bidict = bidict(
        {
            DetectorMode.COUNTING: "counting",
            DetectorMode.ANALOG: "analog",
            DetectorMode.INTERPOLATING: "interpolating",
            DetectorMode.NO_DETECTOR_MODE: "noDetectorMode",
        }
    )

    class TimingMode(IntEnum):
        # the values are the same as in slsdet.timingMode so no bidict table is required
        AUTO_TIMING = 0
        TRIGGER_EXPOSURE = 1

    class DetectorSettings(IntEnum):
        # [G1_HIGHGAIN, G1_LOWGAIN, G2_HIGHCAP_HIGHGAIN, G2_HIGHCAP_LOWGAIN, G2_LOWCAP_HIGHGAIN, G2_LOWCAP_LOWGAIN, G4_HIGHGAIN, G4_LOWGAIN]
        G1_HIGHGAIN = 0
        G1_LOWGAIN = 1
        G2_HIGHCAP_HIGHGAIN = 2
        G2_HIGHCAP_LOWGAIN = 3
        G2_LOWCAP_HIGHGAIN = 4
        G2_LOWCAP_LOWGAIN = 5
        G4_HIGHGAIN = 6
        G4_LOWGAIN = 7

    detectorSettings_bidict = bidict(
        {
            DetectorSettings.G1_HIGHGAIN: detectorSettings.G1_HIGHGAIN,
            DetectorSettings.G1_LOWGAIN: detectorSettings.G1_LOWGAIN,
            DetectorSettings.G2_HIGHCAP_HIGHGAIN: detectorSettings.G2_HIGHCAP_HIGHGAIN,
            DetectorSettings.G2_HIGHCAP_LOWGAIN: detectorSettings.G2_HIGHCAP_LOWGAIN,
            DetectorSettings.G2_LOWCAP_HIGHGAIN: detectorSettings.G2_LOWCAP_HIGHGAIN,
            DetectorSettings.G2_LOWCAP_LOWGAIN: detectorSettings.G2_LOWCAP_LOWGAIN,
            DetectorSettings.G4_HIGHGAIN: detectorSettings.G4_HIGHGAIN,
            DetectorSettings.G4_LOWGAIN: detectorSettings.G4_LOWGAIN,
        }
    )

    status_dict = {
        runStatus.IDLE: DevState.ON,
        runStatus.ERROR: DevState.FAULT,
        runStatus.WAITING: DevState.STANDBY,
        runStatus.RUN_FINISHED: DevState.ON,
        runStatus.TRANSMITTING: DevState.RUNNING,
        runStatus.RUNNING: DevState.RUNNING,
        runStatus.STOPPED: DevState.ON,
    }

    class FrameDiscardPolicy(IntEnum):
        # the values are the same as in slsdet.timingMode so no bidict table is required
        NO_DISCARD = 0
        DISCARD_EMPTY_FRAMES = 1
        DISCARD_PARTIAL_FRAMES = 2

    SLS_RECEIVER_PORT = device_property(
        dtype="str",
        doc="port of the slsReceiver instance, must match the config",
        default_value="1954",
    )
    PROCESSING_RX_IP = device_property(
        dtype="str",
        doc='ip of 10gbe "PC <-> detector" lane of PC, must match the config',
        default_value="192.168.2.200",
    )
    PROCESSING_RX_PORT = device_property(
        dtype="str",
        doc='port for 10gbe "PC <-> detector" lane of PC, must match the config',
        default_value="50003",
    )
    PROCESSING_TX_IP = device_property(
        dtype="str",
        doc="ip for 1gbe lane (lab local network) of PC, must match the config",
        default_value="192.168.1.118",
    )
    PROCESSING_TX_PORT = device_property(
        dtype="str",
        doc="port for 1gbe (lab local network) lane of PC, must match the config",
        default_value="50001",
    )
    PROCESSING_CORES = device_property(
        dtype="str",
        doc="number of cores for zmq on-time processing",
        default_value="20",
    )
    CONFIG_PATH_REAL = device_property(
        dtype="str",
        doc="path for the config file for a real detector",
        default_value="/home/moench/detector/moench_2021.config",
    )
    CONFIG_PATH_VIRTUAL = device_property(
        dtype="str",
        doc="path for the config file for a virtual detector",
        default_value="/home/moench/detector/moench_2021_virtual.config",
    )
    IS_VIRTUAL_DETECTOR = device_property(
        dtype="bool",
        doc="the flag whether a virtual detector need to be used",
        mandatory=True,
    )
    HTTP_HOST_ADDRESS = device_property(
        dtype="str",
        doc="the ip address of detector pc where the captures can be found",
        default_value="http://10.6.5.148",
    )
    HTTP_ROOT_PATH = device_property(
        dtype="str",
        doc="root page for file listing. currently using nginx on moench. check /etc/nginx/nginx.conf",
        default_value="/files",
    )
    ROOT_USERNAME = device_property(
        dtype="str",
        doc="username of root user. required since slsReceiver should be started with root privileges",
        mandatory=True,
    )
    ROOT_PASSWORD = device_property(
        dtype="str",
        doc="password of specified root user. required since slsReceiver should be started with root privileges",
        mandatory=True,
    )
    EXECUTABLES_PATH = device_property(
        dtype="str",
        doc="path of all moench sls executables",
        default_value="/opt/slsDetectorPackage/build/bin/",
    )
    CLEAR_STARTUP = device_property(
        dtype="bool",
        doc="kill old process instances before startup",
        default_value=True,
    )

    exposure = attribute(
        label="exposure",
        dtype="float",
        unit="s",
        format="%6.2e",
        min_value=25e-9,
        max_value=1e2,
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        doc="single frame exposure time",
    )

    delay = attribute(
        label="delay",
        dtype="float",
        unit="s",
        format="%6.2e",
        min_value=0.0,
        max_value=1e2,
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        doc="delay after trigger signal",
    )
    timing_mode = attribute(
        label="trigger mode",
        dtype=TimingMode,
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        doc="[AUTO_TIMING - internal trigger, TRIGGER_EXPOSURE - external]",
    )
    triggers = attribute(
        label="triggers",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        memorized=False,
        hw_memorized=False,
        fisallowed="isWriteAvailable",
        doc="number of triggers for an acquire session",
    )
    filename = attribute(
        label="filename",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        memorized=True,
        hw_memorized=True,
        doc="File name: [filename]_d0_f[sub_file_index]_[acquisition/file_index].raw",
    )
    filepath = attribute(
        label="filepath",
        dtype="str",
        fisallowed="isWriteAvailable",
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        doc="dir where data files will be written",
    )
    fileindex = attribute(
        label="file_index",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        doc="File name: [filename]_d0_f[sub_file_index]_[acquisition/file_index].raw",
    )
    frames = attribute(
        label="number of frames",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        memorized=True,
        hw_memorized=True,
        doc="amount of frames made per acquisition",
    )
    framemode = attribute(
        label="frame mode",
        dtype=FrameMode,
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        doc="framemode of detector [RAW, FRAME, PEDESTAL, NEWPEDESTAL]",
    )
    detectormode = attribute(
        label="detector mode",
        dtype=DetectorMode,
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        doc="detectorMode [COUNTING, ANALOG, INTERPOLATING]",
    )
    filewrite = attribute(
        label="enable or disable file writing",
        dtype="bool",
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        doc="turn of/off writing file to disk",
    )
    highvoltage = attribute(
        display_level=DispLevel.EXPERT,
        label="bias voltage on sensor",
        dtype="int",
        unit="V",
        min_value=60,
        max_value=200,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        access=AttrWriteType.READ_WRITE,
    )
    period = attribute(
        label="period",
        unit="s",
        dtype="float",
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        fisallowed="isWriteAvailable",
        doc="period between acquisitions",
    )
    samples = attribute(
        display_level=DispLevel.EXPERT,
        label="samples amount",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        doc="in analog mode only",
    )
    settings = attribute(
        display_level=DispLevel.EXPERT,
        label="gain settings",
        dtype=DetectorSettings,
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        memorized=True,
        hw_memorized=True,
        doc="[G1_HIGHGAIN, G1_LOWGAIN, G2_HIGHCAP_HIGHGAIN, G2_HIGHCAP_LOWGAIN, G2_LOWCAP_HIGHGAIN, G2_LOWCAP_LOWGAIN, G4_HIGHGAIN, G4_LOWGAIN]",
    )  # converted from enums
    zmqip = attribute(
        display_level=DispLevel.EXPERT,
        label="zmq ip address",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        doc="ip to listen to zmq data streamed out from receiver or intermediate process",
    )
    zmqport = attribute(
        display_level=DispLevel.EXPERT,
        label="zmq port",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        doc="port number to listen to zmq data",
    )  # can be either a single int or list (or tuple) of ints
    rx_discardpolicy = attribute(
        display_level=DispLevel.EXPERT,
        label="discard policy",
        dtype=FrameDiscardPolicy,
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        doc="discard policy of corrupted frames [NO_DISCARD/DISCARD_EMPTY_FRAMES/DISCARD_PARTIAL_FRAMES]",
    )  # converted from enums
    rx_framescaught = attribute(
        display_level=DispLevel.EXPERT,
        label="frames caught",
        dtype="int",
        access=AttrWriteType.READ,
        fisallowed="isWriteAvailable",
        doc="number of frames which were successfully transferred",
    )  # need to be checked, here should be a list of ints
    rx_hostname = attribute(
        display_level=DispLevel.EXPERT,
        label="receiver hostname",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        doc="receiver hostname or IP address",
    )
    rx_tcpport = attribute(
        display_level=DispLevel.EXPERT,
        label="tcp rx_port",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        doc="port for for client-receiver communication via TCP",
    )
    rx_status = attribute(
        display_level=DispLevel.EXPERT,
        label="receiver rx/tx status",
        dtype="str",
        access=AttrWriteType.READ,
        fisallowed="isWriteAvailable",
    )
    detector_status = attribute(
        label="detector tango state",
        dtype="DevState",
        access=AttrWriteType.READ,
    )
    receiver_status = attribute(
        label="receiver tango state",
        dtype="DevState",
        access=AttrWriteType.READ,
    )
    rx_zmqstream = attribute(
        display_level=DispLevel.EXPERT,
        label="data streaming via zmq",
        dtype="bool",
        access=AttrWriteType.READ_WRITE,
        fisallowed="isWriteAvailable",
        doc="enable/disable streaming via zmq",
    )  # will be further required for preview direct from stream
    rx_version = attribute(
        display_level=DispLevel.EXPERT,
        label="rec. version",
        dtype="str",
        access=AttrWriteType.READ,
        fisallowed="isWriteAvailable",
        doc="version of receiver formatatted as [0xYYMMDD]",
    )

    firmware_version = attribute(
        display_level=DispLevel.EXPERT,
        label="det. version",
        dtype="str",
        access=AttrWriteType.READ,
        fisallowed="isWriteAvailable",
        doc="version of detector software",
    )

    raw_detector_status = attribute(
        display_level=DispLevel.EXPERT,
        label="detector status",
        dtype="str",
        access=AttrWriteType.READ,
        doc="raw status of detector",
    )
    tiff_fullpath_next = attribute(
        label="next capture path",
        dtype="str",
        access=AttrWriteType.READ,
        fisallowed="isWriteAvailable",
        doc="full path of the next capture",
    )
    tiff_fullpath_last = attribute(
        label="last capture path",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        doc="full path of the last capture",
    )
    tiff_httppath_last = attribute(
        display_level=DispLevel.EXPERT,
        label="http path",
        dtype="str",
        access=AttrWriteType.READ,
        doc="full http path for the last capture with file",
    )
    tiff_httppath_next = attribute(
        display_level=DispLevel.EXPERT,
        label="http path",
        dtype="str",
        access=AttrWriteType.READ,
        doc="full http path for the next capture with file",
        fisallowed="isWriteAvailable",
    )

    sum_image_last = attribute(
        display_level=DispLevel.EXPERT,
        label="sum last image",
        dtype="DevLong",
        dformat=AttrDataFormat.IMAGE,
        max_dim_x=400,
        max_dim_y=400,
        access=AttrWriteType.READ_WRITE,
        doc="last summarized 400x400 image from detector",
    )

    def init_device(self):
        Device.init_device(self)
        self.set_change_event("sum_image_last", True, False)
        self.set_state(DevState.INIT)
        self.get_device_properties(self.get_device_class())
        MAX_ATTEMPTS = 5
        self.attempts_counter = 0
        computer_setup.kill_all_pc_processes(self.ROOT_USERNAME, self.ROOT_PASSWORD)
        time.sleep(3)
        computer_setup.init_pc(
            virtual=self.IS_VIRTUAL_DETECTOR,
            SLS_RECEIVER_PORT=self.SLS_RECEIVER_PORT,
            PROCESSING_RX_IP=self.PROCESSING_RX_IP,
            PROCESSING_RX_PORT=self.PROCESSING_RX_PORT,
            PROCESSING_TX_IP=self.PROCESSING_TX_IP,
            PROCESSING_TX_PORT=self.PROCESSING_TX_PORT,
            PROCESSING_CORES=self.PROCESSING_CORES,
            CONFIG_PATH_REAL=self.CONFIG_PATH_REAL,
            CONFIG_PATH_VIRTUAL=self.CONFIG_PATH_VIRTUAL,
            EXECUTABLES_PATH=self.EXECUTABLES_PATH,
            ROOT_USERNAME=self.ROOT_USERNAME,
            ROOT_PASSWORD=self.ROOT_PASSWORD,
            CLEAR_STARTUP=self.CLEAR_STARTUP,
        )
        while not computer_setup.is_pc_ready() and self.attempts_counter < MAX_ATTEMPTS:
            time.sleep(0.5)
            self.attempts_counter += 1
        if not computer_setup.is_pc_ready:
            self.delete_device()
            self.info_stream("Unable to start PC")
        self.info_stream("PC is ready")
        self.moench_device = Moench()
        try:
            st = self.moench_device.rx_status
            self.info_stream("Current device status: %s" % st)
            self.set_state(DevState.ON)
        except RuntimeError as e:
            self.set_state(DevState.FAULT)
            self.info_stream("Unable to establish connection with detector\n%s" % e)
            self.delete_device()

    def isWriteAvailable(self, value):
        # slsdet.runStatus.IDLE, ERROR, WAITING, RUN_FINISHED, TRANSMITTING, RUNNING, STOPPED
        if self.moench_device.status in (
            runStatus.IDLE,
            runStatus.WAITING,
            runStatus.STOPPED,
        ):
            return True
        return False

    def read_exposure(self):
        return self.moench_device.exptime

    def write_exposure(self, value):
        self.moench_device.exptime = value

    def read_delay(self):
        return self.moench_device.delay

    def write_delay(self, value):
        self.moench_device.delay = value

    def read_fileindex(self):
        return self.moench_device.findex

    def write_fileindex(self, value):
        self.moench_device.findex = value

    def read_timing_mode(self):
        return self.TimingMode(self.moench_device.timing.value)

    def write_timing_mode(self, value):
        self.moench_device.timing = timingMode(value)

    def read_triggers(self):
        return self.moench_device.triggers

    def write_triggers(self, value):
        self.moench_device.triggers = value

    def read_filename(self):
        return self.moench_device.fname

    def write_filename(self, value):
        self.moench_device.fname = value

    def read_filepath(self):
        return str(self.moench_device.fpath)

    def write_filepath(self, value):
        if not os.path.isdir(value):
            try:
                os.makedirs(value)
            except PermissionError:
                self.error_stream(f"no permission to create a directory in {value}")
            except OSError:
                self.error_stream(f"os error while creating a dir in {value}")
        if os.path.exists(value) & os.path.isdir(value):
            try:
                self.moench_device.fpath = value
            except:
                self.error_stream("not valid filepath")

    def read_frames(self):
        return self.moench_device.frames

    def write_frames(self, value):
        self.moench_device.frames = value

    def read_framemode(self):
        try:
            framemode = self.frameMode_bidict.inverse[
                self.moench_device.rx_jsonpara["frameMode"]
            ]
        except:
            framemode = self.FrameMode.NO_FRAME_MODE
        return framemode

    def write_framemode(self, value):
        self.moench_device.rx_jsonpara["frameMode"] = self.frameMode_bidict[
            self.FrameMode(value)
        ]

    def read_detectormode(self):
        try:
            detectormode = self.detectorMode_bidict.inverse[
                self.moench_device.rx_jsonpara["detectorMode"]
            ]
        except:
            detectormode = self.DetectorMode.NO_DETECTOR_MODE
            self.error_stream("no detectormode set")
        return detectormode

    def write_detectormode(self, value):
        self.moench_device.rx_jsonpara["detectorMode"] = self.detectorMode_bidict[
            self.DetectorMode(value)
        ]

    def read_filewrite(self):
        return self.moench_device.fwrite

    def write_filewrite(self, value):
        self.moench_device.fwrite = value

    def read_highvoltage(self):
        return self.moench_device.highvoltage

    def write_highvoltage(self, value):
        try:
            self.moench_device.highvoltage = value
        except RuntimeError:
            self.error_stream("not allowed highvoltage")

    def read_period(self):
        return self.moench_device.period

    def write_period(self, value):
        self.moench_device.period = value

    def read_samples(self):
        return self.moench_device.samples

    def write_samples(self, value):
        self.moench_device.samples = value

    def read_settings(self):
        return self.detectorSettings_bidict.inverse[self.moench_device.settings]

    def write_settings(self, value):
        self.moench_device.settings = self.detectorSettings_bidict[
            self.DetectorSettings(value)
        ]

    def read_zmqip(self):
        return str(self.moench_device.rx_zmqip)

    def write_zmqip(self, value):
        if bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", value)):
            self.moench_device.rx_zmqip = IpAddr(value)
        else:
            self.error_stream("not valid ip address")

    def read_zmqport(self):
        return self.moench_device.rx_zmqport

    def write_zmqport(self, value):
        self.moench_device.rx_zmqport = value

    def read_rx_discardpolicy(self):
        return self.FrameDiscardPolicy(self.moench_device.rx_discardpolicy.value)

    def write_rx_discardpolicy(self, value):
        self.moench_device.rx_discardpolicy = frameDiscardPolicy(value)

    def read_rx_framescaught(self):
        return self.moench_device.rx_framescaught

    def write_rx_framescaught(self, value):
        pass

    def read_rx_hostname(self):
        return self.moench_device.rx_hostname

    def write_rx_hostname(self, value):
        self.moench_device.rx_hostname = value

    def read_rx_tcpport(self):
        return self.moench_device.rx_tcpport

    def write_rx_tcpport(self, value):
        self.moench_device.rx_tcpport = value

    def read_rx_status(self):
        return str(self.moench_device.rx_status)

    def write_rx_status(self, value):
        pass

    def read_detector_status(self):
        tango_state = self.status_dict.get(self.moench_device.status)
        return tango_state

    def write_detector_status(self, value):
        pass

    def read_receiver_status(self):
        tango_state = self.status_dict.get(self.moench_device.getReceiverStatus()[0])
        return tango_state

    def write_receiver_status(self, value):
        pass

    def read_rx_zmqstream(self):
        return self.moench_device.rx_zmqstream

    def write_rx_zmqstream(self, value):
        self.moench_device.rx_zmqstream = value

    def read_rx_version(self):
        return self.moench_device.rx_version

    def write_rx_version(self, value):
        pass

    def read_firmware_version(self):
        return self.moench_device.firmwareversion

    def write_firmware_version(self, value):
        pass

    def read_tiff_fullpath_next(self):
        # [filename]_d0_f[sub_file_index]_[acquisition/file_index].raw"
        savepath = self.read_filepath()
        filename = self.read_filename()
        file_index = self.read_fileindex()
        postfix = (
            "_ped"
            if self.read_framemode()
            in (self.FrameMode.PEDESTAL, self.FrameMode.NEWPEDESTAL)
            else ""
        )
        fullpath = os.path.join(savepath, f"{filename}_{file_index}{postfix}.tiff")
        return fullpath

    def write_tiff_fullpath_next(self, value):
        pass

    def read_tiff_fullpath_last(self):
        return self._tiff_fullpath_last

    def write_tiff_fullpath_last(self, value):
        self._tiff_fullpath_last = value

    def read_tiff_httppath_last(self):
        http_prefix = self.HTTP_HOST_ADDRESS
        web_prefix = self.HTTP_ROOT_PATH
        http_fullpath = http_prefix + web_prefix + self._tiff_fullpath_last
        return http_fullpath

    def write_tiff_httppath_next(self, value):
        pass

    def read_tiff_httppath_next(self):
        http_prefix = self.HTTP_HOST_ADDRESS
        web_prefix = self.HTTP_ROOT_PATH
        filename = self.read_filename()
        file_index = self.read_fileindex()
        savepath = self.read_filepath()
        fullpath = os.path.join(savepath, f"{filename}_{file_index}.tiff")
        http_fullpath = http_prefix + web_prefix + fullpath
        return http_fullpath

    def write_tiff_httppath_last(self, value):
        pass

    def read_raw_detector_status(self):
        return str(self.moench_device.status)

    def write_raw_detector_status(self):
        pass

    def read_sum_image_last(self):
        return self._last_image

    def write_sum_image_last(self, value):
        pass

    def delete_device(self):
        try:
            computer_setup.deactivate_pc(self.ROOT_USERNAME, self.ROOT_PASSWORD)
            self.info_stream("SlsReceiver or zmq socket processes were killed.")
        except Exception:
            self.info_stream(
                "Unable to kill slsReceiver or zmq socket. Please kill it manually."
            )

    def _block_acquire(self):
        tiff_fullpath_current = self.read_tiff_fullpath_next()
        filewriteEnabled = self.read_filewrite()
        self.moench_device.startReceiver()
        self.info_stream("start receiver")
        self.moench_device.startDetector()
        self.info_stream("start detector")
        # in case detector is stopped we want to leave this section earlier
        # time.sleep(exptime * frames)
        """
        A detector takes a while after startDetector() execution to change its state.
        So if there is no delay after startDetector() and self.get_state() check it's very probable that
        detector will be still in ON mode (even not started to acquire.)
        """
        self.write_tiff_fullpath_last(tiff_fullpath_current)
        if filewriteEnabled:
            self.write_fileindex(self.read_fileindex() + 1)
        time.sleep(0.5)
        while self.read_detector_status() != DevState.ON:
            time.sleep(0.1)
        self.moench_device.stopReceiver()
        self.info_stream("stop receiver")

    async def _async_acquire(self, loop):
        await loop.run_in_executor(None, self._block_acquire)
        loop.run_in_executor(None, self._pending_file)
        # update sum_image_last here

    def _pending_file(self):
        c = 0
        MAX_ATTEMPTS = 15
        isFileReady = False
        while not isFileReady and (c < MAX_ATTEMPTS):
            isFileReady = os.path.isfile(self.read_tiff_fullpath_last())
            time.sleep(0.5)
        if isFileReady:
            self._last_image = imread(self.read_tiff_fullpath_last())
            self.push_change_event(
                "sum_image_last", self.read_sum_image_last(), 400, 400
            )

    @command
    async def start_acquire(self):
        if self.moench_device.status == runStatus.IDLE:
            loop = asyncio.get_event_loop()
            future = loop.create_task(self._async_acquire(loop))
        elif self.moench_device.status == runStatus.RUNNING:
            self.info_stream("Detector is acquiring")
        else:
            self.error_stream("Unable to acquire")

    @command
    def test_push_sum_img_event(self):
        self.push_change_event("sum_image_last", self.read_sum_image_last(), 400, 400)

    @command
    def stop_acquire(self):
        self.moench_device.stop()


if __name__ == "__main__":
    MoenchDetectorControl.run_server()
