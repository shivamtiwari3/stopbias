"""stopbias — does an offline full-duplex stop-latency number survive the telephony transport?

Phase 1 asks the question that can be answered with no API keys and no phone calls: given a
trial whose true stop latency is known by construction, how far does a boundary detector's
estimate move when the audio is carried over a narrowband, log-quantised, lossy channel
instead of read from a 16 kHz file?
"""

__version__ = "0.1.0"
