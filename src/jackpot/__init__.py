from .jackpot import Jackpot
from .manifold import AdversarialManifold
from .direct_model import ModelOperator
from .additional_criteria import AdditionalCriteria
from .singular_solvers import SingularSolver

__all__ = [
    "Jackpot",
    "AdversarialManifold",
    "ModelOperator",
    "AdditionalCriteria",
    "SingularSolver",
]

__version__ = "0.1.3"
