from tango import AttrWriteType, DevState, DevFloat, EncodedAttribute
from tango.server import Device, attribute, command, pipe
from slsdet import Moench, runStatus, timingMode, detectorSettings, frameDiscardPolicy
from _slsdet import IpAddr
import subprocess
import time
import os, socket, sys
import re
import signal
from computer_setup import ComputerSetup
from pathlib import PosixPath


class MoenchDetectorControl(Device):
    polling = 1000
    exposure = attribute(
        label="exposure",
        dtype="float",
        unit="s",
        min_value=0.0,
        max_value=1e2,
        min_warning=1e-6,  # check the smallest exposure when packetloss occurs
        max_warning=0.7e2,  # check too long exposures
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        hw_memorized=True,
        polling_period=polling,
        doc="single frame exposure time",
    )
    timing_mode = attribute(
        label="trigger mode",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        memorized=True,
        doc="AUTO - internal trigger, EXT - external]",
    )  # see property timing in pydetector docs
    triggers = attribute(
        label="triggers",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        doc="number of triggers for an acquire session",
    )
    filename = attribute(
        label="filename",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        doc="File name: [filename]_d0_f[sub_file_index]_[acquisition/file_index].raw",
    )
    filepath = attribute(
        label="filepath", dtype="str", doc="dir where data files will be written"
    )
    frames = attribute(
        label="number of frames",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        doc="amount of frames made per acquisition",
    )
    filewrite = attribute(label="enable or disable file writing", dtype="bool")
    highvoltage = attribute(
        label="high voltage on sensor",
        dtype="int",
        unit="V",
        min_value=60,
        max_value=200,
        min_warning=70,
        max_warning=170,
        access=AttrWriteType.READ_WRITE,
    )
    period = attribute(
        label="period",
        unit="s",
        dtype="float",
        access=AttrWriteType.READ_WRITE,
        doc="period between acquisitions",
    )
    samples = attribute(
        label="samples amount",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        doc="in analog mode only",
    )
    settings = attribute(
        label="gain settings",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        doc="[G1_HIGHGAIN, G1_LOWGAIN, G2_HIGHCAP_HIGHGAIN, G2_HIGHCAP_LOWGAIN, G2_LOWCAP_HIGHGAIN, G2_LOWCAP_LOWGAIN, G4_HIGHGAIN, G4_LOWGAIN]",
    )  # converted from enums
    zmqip = attribute(
        label="zmq ip address",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        doc="ip to listen to zmq data streamed out from receiver or intermediate process",
    )
    zmqport = attribute(
        label="zmq port",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        doc="port number to listen to zmq data",
    )  # can be either a single int or list (or tuple) of ints
    rx_discardpolicy = attribute(
        label="discard policy",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        doc="discard policy of corrupted frames [NO_DISCARD/DISCARD_EMPTY/DISCARD_PARTIAL]",
    )  # converted from enums
    rx_missingpackets = attribute(
        label="missed packets",
        dtype="int",
        access=AttrWriteType.READ,
        doc="number of missing packets for each port in receiver",
    )  # need to be checked, here should be a list of ints
    rx_hostname = attribute(
        label="receiver hostname",
        dtype="str",
        access=AttrWriteType.READ_WRITE,
        doc="receiver hostname or IP address",
    )
    rx_tcpport = attribute(
        label="tcp rx_port",
        dtype="int",
        access=AttrWriteType.READ_WRITE,
        doc="port for for client-receiver communication via TCP",
    )
    rx_status = attribute(
        label="receiver status", dtype="str", access=AttrWriteType.READ
    )
    rx_zmqstream = attribute(
        label="data streaming via zmq",
        dtype="bool",
        access=AttrWriteType.READ_WRITE,
        doc="enable/disable streaming via zmq",
    )  # will be further required for preview direct from stream
    rx_version = attribute(
        label="rec. version",
        dtype="str",
        access=AttrWriteType.READ,
        doc="version of receiver formatatted as [0xYYMMDD]",
    )

    firmware_version = attribute(
        label="det. version",
        dtype="str",
        access=AttrWriteType.READ,
        doc="version of detector software",
    )

    def init_device(self):
        self.pc_util = ComputerSetup()
        self.VIRTUAL = True if "--virtual" in sys.argv else False # check whether server started with "--virtual" flag
        Device.init_device(self)
        self.set_state(DevState.INIT)
        if not self.pc_util.init_pc(virtual=self.VIRTUAL):
            self.set_state(DevState.FAULT)
            self.info_stream(
                "Unable to start slsReceiver or zmq socket. Check firewall process and already running instances."
            )
        self.device = Moench()
        try:
            st = self.device.status
            self.info_stream("Current device status: %s" % st)
        except RuntimeError as e:
            self.set_state(DevState.FAULT)
            self.info_stream("Unable to establish connection with detector\n%s" % e)
            self.delete_device()

    def read_exposure(self):
        return self.device.exptime

    def write_exposure(self, value):
        self.device.exptime = value

    def read_timing_mode(self):
        if self.device.timing == timingMode.AUTO_TIMING:
            return "AUTO"
        elif self.device.timing == timingMode.TRIGGER_EXPOSURE:
            return "EXT"
        else:
            self.info_stream("The timing mode is not assigned correctly.")

    def write_timing_mode(self, value):
        if type(value) == str:
            if value.lower() == "auto":
                self.info_stream("Setting auto timing mode")
                self.device.timing = timingMode.AUTO_TIMING
            elif value.lower() == "ext":
                self.info_stream("Setting external timing mode")
                self.device.timing = timingMode.TRIGGER_EXPOSURE
        else:
            self.info_stream('Timing mode should be "AUTO/EXT" string')

    def read_triggers(self):
        return self.device.triggers

    def write_triggers(self, value):
        self.device.triggers = value

    def read_filename(self):
        return self.device.filename

    def write_filename(self, value):
        self.device.filename = value

    def read_filepath(self):
        return str(self.device.fpath)

    def write_filepath(self, value):
        try:
            self.device.fpath = PosixPath(value)
        except TypeError:
            self.error_stream("not valid filepath")

    def read_frames(self):
        return self.device.frames

    def write_frames(self, value):
        self.device.frames = value

    def read_filewrite(self):
        return self.device.fwrite

    def write_filewrite(self, value):
        self.device.fwrite = value

    def read_highvoltage(self):
        return self.device.highvoltage

    def write_highvoltage(self, value):
        try:
            self.device.highvoltage = value
        except RuntimeError:
            self.error_stream("not allowed highvoltage")

    def read_period(self):
        return self.device.period

    def write_period(self, value):
        self.device.period = value

    def read_samples(self):
        return self.device.samples

    def write_samples(self, value):
        self.device.samples = value

    def read_settings(self):
        return str(self.device.settings)

    def write_settings(self, value):
        settings_dict = {
            "G1_HIGHGAIN": 13,
            "G1_LOWGAIN": 14,
            "G2_HIGHCAP_HIGHGAIN": 15,
            "G2_HIGHCAP_LOWGAIN": 16,
            "G2_LOWCAP_HIGHGAIN": 17,
            "G2_LOWCAP_LOWGAIN": 18,
            "G4_HIGHGAIN": 19,
            "G4_LOWGAIN": 20,
        }
        if value in list(settings_dict.keys()):
            self.device.settings = detectorSettings(settings_dict[value])

    def read_zmqip(self):
        return str(self.device.rx_zmqip)

    def write_zmqip(self, value):
        if bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", value)):
            self.device.rx_zmqip = IpAddr(value)
        else:
            self.error_stream("not valid ip address")

    def read_zmqport(self):
        return self.device.rx_zmqport

    def write_zmqport(self, value):
        self.device.rx_zmqport = value

    def read_rx_discardpolicy(self):
        return self.device.rx_discardpolicy

    def write_rx_discardpolicy(self, value):
        disard_dict = {
            "NO_DISCARD": 0,
            "DISCARD_EMPTY_FRAMES": 1,
            "DISCARD_PARTIAL_FRAMES": 2,
        }
        if value in list(disard_dict.keys()):
            self.device.rx_discardpolicy = frameDiscardPolicy(disard_dict[value])

    def read_rx_missingpackets(self):
        return str(self.device.rx_missingpackets)

    def write_rx_missingpackets(self, value):
        pass

    def read_rx_hostname(self):
        return self.device.rx_hostname

    def write_rx_hostname(self, value):
        self.device.rx_hostname = value

    def read_rx_tcpport(self):
        return self.device.rx_tcpport

    def write_rx_tcpport(self, value):
        self.device.rx_tcpport = value

    def read_rx_status(self):
        return str(self.device.rx_status)

    def write_rx_status(self, value):
        pass

    def read_rx_zmqstream(self):
        return self.device.rx_zmqstream

    def write_rx_zmqstream(self, value):
        self.device.rx_zmqstream = value

    def read_rx_version(self):
        return self.device.rx_version

    def write_rx_version(self, value):
        pass

    def read_firmware_version(self):
        return self.device.firmwareversion

    def write_firmware_version(self, value):
        pass

    @command
    def delete_device(self):
        try:
            self.pc_util.deactivate_pc(self.VIRTUAL)
            self.info_stream("SlsReceiver or zmq socket processes were killed.")
        except Exception:
            self.info_stream(
                "Unable to kill slsReceiver or zmq socket. Please kill it manually."
            )

    @command
    def start(self):
        self.device.start()

    @command
    def rx_start(self):
        self.device.rx_start()

    @command
    def rx_stop(self):
        self.device.rx_stop()


if __name__ == "__main__":
    MoenchDetectorControl.run_server()