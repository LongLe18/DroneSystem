from seq.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/phuong/New Volume1/drone/DroneSystem/data/got10k_lmdb'
    settings.got10k_path = '/media/phuong/New Volume1/drone/DroneSystem/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/media/phuong/New Volume1/drone/DroneSystem/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/phuong/New Volume1/drone/DroneSystem/data/lasot_lmdb'
    settings.itb_path = '/media/phuong/New Volume1/drone/DroneSystem/data/itb'
    settings.lasot_path = '/media/phuong/New Volume1/drone/DroneSystem/data/lasot'
    settings.network_path = '/media/phuong/New Volume1/drone/DroneSystem/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/phuong/New Volume1/drone/DroneSystem/data/nfs'
    settings.otb_path = '/media/phuong/New Volume1/drone/DroneSystem/data/OTB2015'
    settings.prj_dir = '/media/phuong/New Volume1/drone/DroneSystem'
    settings.result_plot_path = '/media/phuong/New Volume1/drone/DroneSystem/test/result_plots'
    settings.results_path = '/media/phuong/New Volume1/drone/DroneSystem/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/media/phuong/New Volume1/drone/DroneSystem'
    settings.segmentation_path = '/media/phuong/New Volume1/drone/DroneSystem/test/segmentation_results'
    settings.tc128_path = '/media/phuong/New Volume1/drone/DroneSystem/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/phuong/New Volume1/drone/DroneSystem/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/phuong/New Volume1/drone/DroneSystem/data/trackingnet'
    settings.uav_path = '/media/phuong/New Volume1/drone/DroneSystem/data/UAV123'
    settings.vot_path = '/media/phuong/New Volume1/drone/DroneSystem/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.show_result = True
    return settings

