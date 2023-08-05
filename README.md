# DroneSystem

Drone System includes Object Detection and Object Tracking

Run the following command to set paths for this project
```
python tracking/init/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
tracking/grm/local.py  # paths about training
tracking/seq/local.py  # paths about testing
```
