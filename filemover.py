#copy all folders named task-1 which are located at _outputs/hepco_v2.9_iid_cutoff/IMBALANCEDOMAINNET/10-task/vit/l2p_multi-layer_client_*/models/repeat-1/ to _outputs/hepco_v3.0_iid_cutoff/IMBALANCEDOMAINNET/10-task/vit/l2p_multi-layer_client_*/models/repeat-1/ where hepco_v3.0_iid_cutoff is the new folder name which does not exist yet

import os
import shutil
from tqdm import tqdm

def move_files(source, destination):
    src_folder = os.path.join(source, 'task-1')
    dst_folder = os.path.join(destination, 'task-1')
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for file_name in os.listdir(src_folder):
        src_file = os.path.join(src_folder, file_name)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_folder)
        
def main():
    source = '_outputs/hepco_v6.0_DOMAINNET_iid_cutoff_cutratio_0.4_seed_1/IMBALANCEDOMAINNET/69-task/vit/l2p_multi-layer_server_'
    destination = '_outputs/hepco_v6.0_DOMAINNET_iid_cutoff_cutratio_0.4_seed_1_morehistory/IMBALANCEDOMAINNET/69-task/vit/l2p_multi-layer_server_'
    for i in tqdm(range(10), desc='Moving files',total=10):
        src = source + str(i) + '/models/repeat-2/'
        dst = destination + str(i) + '/models/repeat-2/'
        move_files(src, dst)

if __name__ == '__main__':
    main()