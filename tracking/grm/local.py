import os

from grm.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here

    settings.davis_dir = ''
    settings.got10k_path = os.path.expanduser('~') + '/track/data/GOT10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/media/phuong/New Volume/drone/DroneSystem/data/ITB'
    settings.lasot_path = '/media/phuong/New Volume/drone/DroneSystem/data/LaSOT'
    settings.network_path = '/media/phuong/New Volume/drone/DroneSystem/output/test/networks'  # Where tracking networks are stored
    settings.nfs_path = os.path.expanduser('~') + '/track/data/NFS30'
    settings.otb_path = os.path.expanduser('~') + '/track/data/OTB100'
    settings.prj_dir = '/media/phuong/New Volume/drone/DroneSystem/'
    settings.result_plot_path = '/media/phuong/New Volume/drone/DroneSystem/output/test/result_plots'
    settings.results_path = '/media/phuong/New Volume/drone/DroneSystem/output/test/tracking_results'  # Where to store tracking results
    settings.save_dir = '/media/phuong/New Volume/drone/DroneSystem/'
    settings.segmentation_path = os.path.expanduser('~') + '/track/code/GRM/output/test/segmentation_results'
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = os.path.expanduser('~') + '/track/data/TrackingNet'
    settings.uav_path = os.path.expanduser('~') + '/track/data/UAV123'
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.avist_path = os.path.expanduser('~') + '/track/data/AVisT'
    settings.youtubevos_dir = ''
    settings.show_result = True
    return settings
