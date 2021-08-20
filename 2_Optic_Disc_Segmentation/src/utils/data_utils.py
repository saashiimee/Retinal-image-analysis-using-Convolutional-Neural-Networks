import os
import shutil

import h5py


def mkdir_if_not_exist(dir_name, is_delete=False):
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Directory "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Directory "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["image"][()]
