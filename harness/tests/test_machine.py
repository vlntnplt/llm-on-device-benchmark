"""dmidecode parsing: channels count distinct (controller, channel) pairs —
per-controller "ChannelA"s are different channels, not one."""

from bench.machine import _parse_dimms

DEVICE = """
Memory Device
\tSize: 8 GB
\tLocator: {locator}
\tBank Locator: BANK 0
\tSpeed: 4800 MT/s
\tConfigured Memory Speed: 4800 MT/s
\tRank: 1
"""


def _channels(*locators):
    dimms = _parse_dimms("".join(DEVICE.format(locator=loc) for loc in locators))
    return len({d["_channel"] for d in dimms})


def test_per_controller_channels_are_distinct():
    assert _channels("Controller0-ChannelA-DIMM0", "Controller1-ChannelA-DIMM0") == 2


def test_same_controller_same_channel_collapses():
    assert _channels("ChannelA-DIMM0", "ChannelA-DIMM1") == 1


def test_two_plain_channels():
    assert _channels("ChannelA-DIMM0", "ChannelB-DIMM0") == 2
