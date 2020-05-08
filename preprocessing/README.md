# Pre-processing

Start with the raw data from `raw.tar`.  The scripts in this folder will remove the missing images, those which will not open, and those which are duplicates.  These should be run in order.

Example
- `python3 1_delete_zombies.py --data=/path/to/raw`
- `python3 2_delete_missing.py --data=/path/to/raw`
- `python3 3_delete_duplicates.py --data=/path/to/raw`
- `python3 4_delete_empty_folders.py --data=/path/to/raw`
