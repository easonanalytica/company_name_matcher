import os

def setup_environment():
    """Set up all necessary environment variables for the package."""
    # OpenMP settings
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['KMP_INIT_AT_FORK']='FALSE'
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer parallelism warning