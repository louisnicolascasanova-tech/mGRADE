"""
Training Package for Den-minGRU

Modular training utilities including:
- optimizers: Learning rate schedules and train state creation
- train_loop: Training loop and gradient updates
- evaluation: Validation and evaluation functions
- inference: Inference-specific utilities

Public API maintains backward compatibility with original training.py
"""

# ============================================================================
# Optimizer and Training State Creation
# ============================================================================
from .optimizers import (
    create_learning_rate_fn,
    create_learning_rate_map,
    create_train_state,
    init_model,
    map_nested_fn,
)

# ============================================================================
# Training Loop
# ============================================================================
from .train_loop import (
    run_epoch,
    apply_model,
    apply_retrieval_model,
    update_model,
    clip_gradients_elementwise,
)

# ============================================================================
# Evaluation
# ============================================================================
from .evaluation import (
    eval_model,
    validate,
)

# ============================================================================
# Inference
# ============================================================================
from .inference import (
    inf_model,
)

__all__ = [
    # Optimizers
    'create_learning_rate_fn',
    'create_learning_rate_map',
    'create_train_state',
    'init_model',
    'map_nested_fn',

    # Training
    'run_epoch',
    'apply_model',
    'apply_retrieval_model',
    'update_model',
    'clip_gradients_elementwise',

    # Evaluation
    'eval_model',
    'validate',

    # Inference
    'inf_model',
]
