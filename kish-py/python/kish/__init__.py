# Re-export all types from the native module
from .kish import Team, Square, GameStatus, Action, Board, Game

__all__ = ["Team", "Square", "GameStatus", "Action", "Board", "Game"]
from importlib.metadata import version as _dist_version

__version__ = _dist_version("kish")
