from seq.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\got10k_lmdb'
    settings.got10k_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\lasot_extension_subset'
    settings.lasot_lmdb_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\lasot_lmdb'
    settings.lasot_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\lasot'
    settings.network_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\output\\test/networks'    # Where tracking networks are stored.
    settings.nfs_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\\nfs'
    settings.otb_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\OTB2015'
    settings.prj_dir = 'D:\PracticePY\Project\Drone-2023\DroneSystem'
    settings.result_plot_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\output\\test/result_plots'
    settings.results_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\output\\test/tracking_results'    # Where to store tracking results
    settings.save_dir = 'D:\PracticePY\Project\Drone-2023\DroneSystem\output'
    settings.segmentation_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\output\\test/segmentation_results'
    settings.tc128_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\\tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\\trackingnet'
    settings.uav_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\UAV123'
    settings.vot_path = 'D:\PracticePY\Project\Drone-2023\DroneSystem\data\VOT2019'
    settings.youtubevos_dir = ''

    return settings

