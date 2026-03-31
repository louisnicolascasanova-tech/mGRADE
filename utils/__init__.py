"""
Utils package for Den-minGRU project.
Contains data loading utilities, logging utilities, and helper functions.
"""

from .data import (
    # Stratified Sampling Utilities
    StratifiedBatchSampler,
    create_stratified_dataloader,
    extract_labels_from_dataset,

    # Dataset Creation Functions
    create_mnist_classification_dataset,
    create_cifar_gs_classification_dataset,
    create_lra_imdb_classification_dataset,
    create_lra_listops_classification_dataset,
    create_lra_aan_classification_dataset,
    create_lra_path32_classification_dataset,
    create_lra_pathx_classification_dataset,
    create_speechcommands35_classification_dataset,
    create_uea_classification_dataset,

    # Helper Functions
    write_config_yaml,
    make_data_loader,
    prep_batch,
    setup_random_seeds,
    parse_experiment_config,
    generate_experiment_id,
    create_experiment_directories,
    compute_class_weights,

    # Classes
    SpeechCommandsDataset,

    # Constants
    PX,
    DEFAULT_CACHE_DIR_ROOT,
)

from .logging import (
    # Logging Configuration
    LoggingConfig,

    # Logging Functions
    log_training_batch,
    log_classification_metrics,
    log_gradient_histograms,
    log_network_dynamics,
    log_monitor_data,
    log_parameter_matrices,
    log_batch_metrics,

    # Helper Functions
    create_histogram_and_stats,
    create_confusion_matrix_plot,
)

from .types import (
    # Type Definitions
    DatasetInfo,
    ExperimentDirs,
    TrainingState,
    TrainingResults,
)

from .pipeline import (
    # Pipeline Functions
    setup_dataset,
    create_model,
    setup_training,
    setup_experiment_dirs,
    train_model,

    # Helper Functions
    tabulate_model,
    print_dcls_parameters,
    run_validation_for_dataset,
    update_improvement_threshold,
    check_early_stopping,
)

__all__ = [
    # Stratified Sampling
    'StratifiedBatchSampler',
    'create_stratified_dataloader',
    'extract_labels_from_dataset',

    # Dataset Creation
    'create_mnist_classification_dataset',
    'create_cifar_gs_classification_dataset',
    'create_lra_imdb_classification_dataset',
    'create_lra_listops_classification_dataset',
    'create_lra_aan_classification_dataset',
    'create_lra_path32_classification_dataset',
    'create_lra_pathx_classification_dataset',
    'create_speechcommands35_classification_dataset',
    'create_uea_classification_dataset',

    # Helpers
    'write_config_yaml',
    'make_data_loader',
    'prep_batch',
    'setup_random_seeds',
    'parse_experiment_config',
    'generate_experiment_id',
    'create_experiment_directories',
    'compute_class_weights',

    # Classes
    'SpeechCommandsDataset',

    # Constants
    'PX',
    'DEFAULT_CACHE_DIR_ROOT',

    # Logging Configuration
    'LoggingConfig',

    # Logging Functions
    'log_training_batch',
    'log_classification_metrics',
    'log_gradient_histograms',
    'log_network_dynamics',
    'log_monitor_data',
    'log_parameter_matrices',
    'log_batch_metrics',
    'create_histogram_and_stats',
    'create_confusion_matrix_plot',

    # Type Definitions
    'DatasetInfo',
    'ExperimentDirs',
    'TrainingState',
    'TrainingResults',

    # Pipeline Functions
    'setup_dataset',
    'create_model',
    'setup_training',
    'setup_experiment_dirs',
    'train_model',
    'tabulate_model',
    'print_dcls_parameters',
    'run_validation_for_dataset',
    'update_improvement_threshold',
    'check_early_stopping',
]
