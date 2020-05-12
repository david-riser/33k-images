import argparse
import os

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=str)
    ap.add_argument('--output_dir', required=True, type=str)
    return ap.parse_args()

if __name__ == "__main__":

    args = get_args()
    
    processors = ['1_delete_zombies.py', '2_delete_missing.py', '3_delete_duplicates.py',
                  '4_delete_empty_folders.py', '5_create_file_lists.py']

    print("[INFO] Running cleaning pipeline! ")
    for processor in processors:
        print("[INFO] Running pipeline process {}".format(processor))

        if '5' in processor:
            command = 'python3 {} --data={} --output_dir={}'.format(
                processor, args.data, args.output_dir)
        else:
            command = 'python3 {} --data={}'.format(
                processor, args.data)
            
        os.system(command)

    print("[INFO] Done cleaning! ")
