import pickle
import h5py

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


"""
Due to the possibility of failing to read a region based on the coord topleft coordinates when extracting features, 
some patch features are set and marked with coord as (-1, -1); they are all saved in the h5 file corresponding to a single slide.
read the folder where the h5 files are located, and then count the ideal number of patches and the actual number of patches with extracted features for the entire queue.
"""
def stat_feat_patch_num(feat_dir, to_csv = False):
    import os

    h5filelist = os.listdir(os.path.join(feat_dir, 'h5_files'))
    h5filelist = sorted(h5filelist)

    import pandas as pd
    import numpy as np

    total = len(h5filelist)
    default_df_dict = {'slide_id': np.full((total), 'to_be_added'),
                       'num_patch_coord': np.full((total), 1, dtype=np.uint32),
                       'num_patch_feats': np.full((total), 1, dtype=np.uint32)}
    # default_df_dict.update({'label': np.full((total), -1)})
    stat_slides = pd.DataFrame(default_df_dict)

    print_patcherror_slidelist = []
    for idx, filename in enumerate(h5filelist):
        print(filename)

        h5file_path = os.path.join(feat_dir, 'h5_files', filename)
        file = h5py.File(h5file_path, "r")

        num_patch_coord = len(file['coords'])
        print('(Ideal) Num of patch: ', num_patch_coord)
        num_patch_feats = sum(file['coords'][:, 0] != -1)
        print('(Actual) Num of patch: ', num_patch_feats)
        
        stat_slides.loc[idx, 'slide_id'] = filename.split('.')[0]
        stat_slides.loc[idx, 'num_patch_coord'] = num_patch_coord
        stat_slides.loc[idx, 'num_patch_feats'] = num_patch_feats
        
        if num_patch_feats != num_patch_coord:
            print_patcherror_slidelist.append(np.array([filename, num_patch_coord, num_patch_feats]))
    
    print(np.array(print_patcherror_slidelist))
    
    if to_csv:
        stat_slides.to_csv(os.path.join(feat_dir, 'slides_of_num_patch_feat_autogen.csv'), index=False)


if __name__ == "__main__":
    stat_feat_patch_num(feat_dir="test", to_csv=True)
