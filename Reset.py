import os
import shutil

# Function to reset the environment
def reset_history(airfoil_history_dir,cl_history_path):
    """Reset the airfoil history directory and cl_history directory."""
    # Safely remove and recreate the airfoil history directory
    if os.path.exists(airfoil_history_dir):
        try:
            shutil.rmtree(airfoil_history_dir)
            print(f"Deleted {airfoil_history_dir} directory and all its contents")
        except Exception as e:
            print(f"Error deleting {airfoil_history_dir}: {e}")

    os.makedirs(airfoil_history_dir, exist_ok=True)

    # Safely handle cl_history as either a file or a directory
    if os.path.exists(cl_history_path):
        try:
            if os.path.isdir(cl_history_path):
                shutil.rmtree(cl_history_path)
                print(f"Deleted {cl_history_path} directory and all its contents")
                os.makedirs(cl_history_path, exist_ok=True)
            else:
                os.remove(cl_history_path)
                print(f"Deleted {cl_history_path} file")
        except Exception as e:
            print(f"Error deleting {cl_history_path}: {e}")
    else:
        os.makedirs(cl_history_path, exist_ok=True)
