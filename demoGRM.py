import argparse
import os
import sys
import warnings

import torch

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

warnings.filterwarnings('ignore')

from tracking.grm.mainTracker import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, video_path=None):
    """
    Run tracker on sequence or dataset.

    Args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run ID.
        dataset_name: Name of dataset (otb, nfs, uav, trackingnet, got_test, got_val, lasot, lasot_ext).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """
    trackers = Tracker(tracker_name, tracker_param, None, run_id)

    trackers.run_video(video_path)


def main():
    parser = argparse.ArgumentParser(description='run tracker on sequence or dataset')
    parser.add_argument('--tracker', type=str, default='grm', help='name of tracking method')
    parser.add_argument('--param', type=str, default='vitb_256_ep300', help='name of config file')
    parser.add_argument('--id', type=int, default=None, help='the run ID')
    parser.add_argument('--video_path', type=str, default=None, help='sequence number or name')
    parser.add_argument('--debug', type=int, default=0, help='debug level')
    parser.add_argument('--threads', type=int, default=0, help='number of threads')
    parser.add_argument('--num_gpus', type=int, default=8, help='num of GPUs you want to use')
    parser.add_argument('--num_cpus', type=int, default=8, help='num of CPUs you want to use')

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = str(args.num_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_cpus)
    os.environ['MKL_NUM_THREADS'] = str(args.num_cpus)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.num_cpus)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_cpus)
    torch.set_num_threads(args.num_cpus)

    run_tracker(args.tracker, args.param, args.id, args.video_path)


if __name__ == '__main__':
    main()
