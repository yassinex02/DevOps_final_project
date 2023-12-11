#%%
import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath, topdown=True):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]  # Add any other directories you want to exclude
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

# Example usage
directory_to_explore = '.'  # Replace '.' with the path of the directory you want to explore
list_files(directory_to_explore)

# %%
