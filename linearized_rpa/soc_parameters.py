from dataclasses import dataclass


@dataclass
class SOCParameters:
    """Spin-orbit coupling parameters [eV]."""

    rashba: float = 0
    valley_zeeman: float = 0
