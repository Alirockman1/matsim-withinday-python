import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import os

# If using a shared cluster file system,
def initialize_shared_database(db_file_path, study_name):
    """
    Validates existence of the sqlite schema file on shared storage, initializing it if absent.
    """
    db_dir = os.path.dirname(db_file_path)
    os.makedirs(db_dir, exist_ok=True)
    
    file_storage = JournalFileStorage(db_file_path)
    storage_engine = JournalStorage(file_storage)
        
    # Automatically intercepts and sets up the schema tables safely if file is empty
    optuna.create_study(
        study_name=study_name,
        storage=storage_engine,
        directions=[
            "maximize",
            "minimize",
            "minimize"
        ],
        load_if_exists=True
    )

    print("Study initialized successfully.")
    
    return storage_engine
