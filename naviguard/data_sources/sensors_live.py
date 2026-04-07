"""Stub for future live-sensor integration (real IP cameras, AIS receivers).

When real hardware becomes available, replace the placeholder implementations
below with drivers that connect to physical devices while keeping the same
interface used by the rest of the NaviGuard pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from naviguard.data_sources.navisense_client import (
    NaviSenseClientBase,
    OwnShipState,
    SyntheticAISTrack,
)

logger = logging.getLogger(__name__)


class LiveCameraClient:
    """Read frames from a live IP camera (RTSP / HTTP MJPEG).

    Parameters
    ----------
    url : str
        Camera stream URL, e.g. ``rtsp://192.168.1.100:554/stream``
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._cap = None

    def open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera stream: {self.url}")
        logger.info("Opened live camera stream: %s", self.url)

    def get_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class LiveAISParser:
    """Read AIS messages from a real NMEA/TCP feed.

    Parameters
    ----------
    host : str
        AIS receiver host (e.g. an AIS multiplexer or SignalK server).
    port : int
        TCP port for NMEA sentences.
    """

    def __init__(self, host: str = "localhost", port: int = 10110) -> None:
        self.host = host
        self.port = port
        self._socket = None

    def connect(self) -> None:
        import socket

        self._socket = socket.create_connection((self.host, self.port), timeout=5)
        logger.info("Connected to AIS feed at %s:%d", self.host, self.port)

    def poll(self) -> List[SyntheticAISTrack]:
        """Read and parse one batch of NMEA sentences.

        Returns a (possibly empty) list of :class:`SyntheticAISTrack` objects.
        Real implementation would use a library such as ``pyais`` to parse
        ``!AIVDM`` sentences into vessel records.
        """
        # TODO: replace with full NMEA / pyais parsing implementation
        logger.debug("LiveAISParser.poll() – stub, returning empty list")
        return []

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class LiveGPSReader:
    """Read own-ship position from a GPS/GNSS unit over NMEA serial.

    Parameters
    ----------
    port : str
        Serial port, e.g. ``/dev/ttyUSB0`` on Linux or ``COM3`` on Windows.
    baud : int
        Baud rate (default 4800 for standard NMEA 0183).
    """

    def __init__(self, port: str, baud: int = 4800) -> None:
        self.port = port
        self.baud = baud
        self._serial = None

    def open(self) -> None:
        import serial  # type: ignore[import]

        self._serial = serial.Serial(self.port, self.baud, timeout=1)
        logger.info("Opened GPS serial port %s @ %d baud", self.port, self.baud)

    def get_own_ship(self) -> OwnShipState:
        """Parse the next GNSS fix from the serial buffer.

        Returns an :class:`OwnShipState` with the current position.
        Speed/heading defaults to 0 if not yet available.
        """
        # TODO: replace with full NMEA GGA/RMC parsing (e.g. via pynmea2)
        logger.debug("LiveGPSReader.get_own_ship() – stub")
        return OwnShipState(timestamp=time.time())

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
