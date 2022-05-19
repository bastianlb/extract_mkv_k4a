import argparse
from glob import glob
import subprocess
import os
import numpy as np
import csv
from tqdm import tqdm
import re

"""
This script checks whether mkv files contain MIDI Timecodes
Outputs a CSV file with filename and has_midi_timecode as columns
"""


def parse_args():
    """
    Parses command line arguments for input directory

    Returns:
        args: the arguments
    """    
    
    parser = argparse.ArgumentParser(description="Check what MKV Files contain MIDI Timecodes")
    parser.add_argument(
        '--indir', help='Directory of all the recordings. Script automatically gatheres all mkv files under a cn04 directory', 
        type=str, default='/mnt/',
    )
    parser.add_argument(
        '--outdir', help='Directory where we save the csv file', type=str, default='output/',
    )
    parser.add_argument(
        '--file_name', help='Name of the output file', type=str, default='midi_timecodes.csv'
    )
    parser.add_argument(
        '--filter_output', help='Whether to filter the result', type=bool, default=True
    )

    args = parser.parse_args()
    return args


def check_for_midi(files):
    result = np.full(len(files), 0)
    print("Checking every mkv file...")
    for i, file in enumerate(tqdm(files)):
        bashCommand = f'mkvinfo {file} | grep TIMECODE'
        process = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        output, error = process.communicate()
        if error != None:
            raise RuntimeError(error)
        if output != '':
            result[i] = 1
    return result


def create_output(files, data, output_dir, file_name):
    if not os.path.isdir(output_dir):
        print("--- Creating output directory")
        os.mkdir(output_dir)
    else:
        print("--- Output directory already exists")


    path = os.path.join(output_dir, file_name)

    if os.path.exists(path):
        os.remove(path)
        print("--- Deleted old csv file")

    header = ['file', 'has_midi_timecode']

    with open(path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerows(data)

    print("--- Finished writing the csv file in:", path)

    
def get_mkv_files(input_dir, output_dir, file_name, filter_output):
    # All mkv files in camera node 4
    files = sorted(glob(os.path.join(input_dir, '**/cn04/*.mkv'), recursive=True))
    files = list(filter(lambda file: 'calibrations' not in file.split('/'), files))


    if filter_output:
        filtered_files = []
        file_name = file_name[:-4] + '_filtered.csv'
        recordings = set()
        r = re.compile('.*_animal_trial_.*')
        for file in files:
            dirs = file.split('/')
            recording = list(filter(r.match, dirs))[0]
            if os.path.basename(file) == 'capture-000000.mkv' and recording not in recordings:
                recordings.add(recording) 
                filtered_files.append(file)
        files = filtered_files
            


    result = check_for_midi(files)
    assert (len(files) == len(result))

    # make data for csv writer
    data = []
    for i in range(len(files)):
        data.append([files[i], result[i]])

    # Write as debug statistics
    print(f'--- Out of {len(result)} mkv files, {sum(result)} files have MIDI timecoding')

    create_output(files, data, output_dir, file_name)
    

def print_cfgs(args):
    print(f"""Your Configurations:
    Input Directory: {args.indir}
    Output Directory: {args.outdir}
    File Name: {args.file_name}
        """
    )


def main():
    args = parse_args()
    print_cfgs(args)
    get_mkv_files(args.indir, args.outdir, args.file_name, args.filter_output)
    

if __name__ == '__main__':
    main()
    