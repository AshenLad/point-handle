import string
from os.path import split
import numpy as np
from numpy.ma.core import array, shape
from pypcd4 import pypcd4, PointCloud, Encoding
import os
import shutil


file_folder = './labeled'
files = os.listdir(file_folder)
# print(len(file))

#删除文件名中的空格
for file in files:
    if ' ' in file:
        old_path = os.path.join(file_folder, file)
        new_path = old_path.replace(' ', '')
        os.rename(old_path, new_path)
        print('改名成功')

files = os.listdir(file_folder)
for file in files:
    try:
        path = os.path.join(file_folder, file)
        pcd_data : PointCloud = PointCloud.from_path(path)

        table_head = dict()
        #获得pcd中各个属性的位置
        for key,val in enumerate(pcd_data.fields):
            table_head[val] = key
        # print(table_head)

        pc_array = pcd_data.numpy()
        point = np.zeros((pc_array.shape[0], 5))
        # print(point.shape)

        for i in range(pc_array.shape[0]):
            point[i][0] = pc_array[i][table_head['x']]
            point[i][1] = pc_array[i][table_head['y']]
            point[i][2] = pc_array[i][table_head['z']]
            point[i][3] = pc_array[i][table_head['intensity']]
            point[i][4] = pc_array[i][table_head['label']]

            # 防止有的人将点云中的其他部分标签标注为3
            if point[i][4] == 3:
                point[i][4] = 0

        # print(point)
        pcd = PointCloud.from_xyzil_points(point, label_type=np.uint32)
        print(pcd)
        file_name = path.split('/')[-1]
        save_path = os.path.join('./finsh', file_name)
        pcd.save(save_path, encoding=Encoding.ASCII)
        print('pcd文件保存成功')

        #已转换文件移动
        if not os.path.exists('useless'):
            os.makedirs('useless')
        shutil.move(path, 'useless')
        print('已转换文件移动')

    except KeyError as e:
        #个别文件存在标注错误，都是邢政一的问题
        #Scalar_field_#4是label
        #Scalar_field_#3是time
        #Scalar_field_#2是ring
        #Scalar_field是intensity

        # 有问题的文件先单独存放,之后用文本的方式处理
        if not os.path.exists('Problematic'):
            os.makedirs('Problematic')
        shutil.move(os.path.join(file_folder, file), 'Problematic')
        print('问题文件转移')
    except FileNotFoundError as e:
        pass
    except Exception as e:
        print('未知问题')
    finally:
        pass


