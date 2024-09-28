from tkinter.tix import ASCII

from numpy.ma.core import append
from pypcd4 import PointCloud, Encoding
import numpy as np
import os
import shutil

def main ():
    os.chdir('./Problematic/txt')
    list_dir = os.listdir('.')
    # print(len(list_dir))

    txt_file = []
    for file in list_dir:
        if(file.split('.')[-1] == 'txt'):
            txt_file.append(file)

    for file in txt_file:
        point_arr = []
        with open(file, 'r') as f:
            for line in f:
                point_arr.append(line.split(' '))

        point_nparr = np.array(point_arr)
        point_nparr = point_nparr.astype('float32')
        point_nparr = np.delete(point_nparr, (4, 5), axis=1)

        pcd = PointCloud.from_xyzil_points(point_nparr, label_type=np.uint32)
        save_name = file.split('.')[:-1]
        save_name = save_name[0] + '.' + save_name[1] + '.pcd'
        save_path = os.path.join('../txt_finsh', save_name)
        pcd.save(save_path, encoding=Encoding.ASCII)



if __name__ == '__main__':
    main()