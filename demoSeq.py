import argparse

from tracking.seq.mainTracker import Tracker

def run_tracker(tracker_name, tracker_param, run_id=None, videp_path=None, debug=0):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset.
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    trackers = Tracker(tracker_name, tracker_param, None, run_id)

    trackers.run_video(videp_path)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--video_path', type=str, default=None, help='path video for run tracker')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')

    args = parser.parse_args()


    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.video_path, args.debug)


if __name__ == '__main__':
    main()
