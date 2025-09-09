from .jackpot import Jackpot
from .adversarial import AdversarialManifold
from .direct_model import ModelOperator
from .stop_criteria import AdditionalCriteria
from .singular_solvers import SingularSolver, singular_vectors

__all__ = [
    "Jackpot",
    "AdversarialManifold",
    "ModelOperator",
    "AdditionalCriteria",
    "SingularSolver",
    "singular_vectors",
]

__version__ = "0.1.2"
