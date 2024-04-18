import os, random
from PIL import Image
from tqdm import tqdm

def read_file(path):
    with open(path, 'r') as f:
        filenames = f.readlines()

    return filenames

def open_image(image_path):
    try:
        image = Image.open(image_path)
        return image
    except:
        return image_path

def match_image(base, time_stemp, filenames_file):
    files = []
    filenames = read_file(filenames_file)
    for filename in filenames:
        cur_file = filename.split()[0]
        files.append(cur_file)
    print(len(files))

    cnt = 0
    for time in time_stemp:
        cur_time = os.path.join(base, time)
        total_data = os.listdir(cur_time)
        for data in total_data:
            cur_data = os.path.join(cur_time, data)
            cur_data = cur_data + '/image_02/data/'
            images = os.listdir(cur_data)
            for image in images:
                cur_image = os.path.join(cur_data, image).replace(base, '')
                if cur_image in files:
                    cnt += 1
    print(cnt)

def match_depth(filenames_file, gt_path):
    filesnames = read_file(filenames_file)
    depths = []

    for filename in filesnames:
        cur_depth = filename.split()[1]
        depths.append(cur_depth)
    print('file depth num:', len(depths))

    cnt = 0
    not_match = []
    depth_files = os.listdir(gt_path)
    for file in depth_files:
        cur_file = os.path.join(gt_path, file)
        cur_file += '/proj_depth/groundtruth/image_02/'
        total_depth = os.listdir(cur_file)
        for depth in total_depth:
            cur_depth = os.path.join(cur_file, depth)
            cur_depth = cur_depth.replace(gt_path, '')
            if cur_depth in depths:
                depths.remove(cur_depth)
    print(len(depths))
    with open('not_match_eval.txt', 'w') as f:
        for a in depths:
            f.write(a + '\n')





if __name__ == '__main__':
    a = '2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000116.png ' \
        '2011_09_26_drive_0057_sync/proj_depth/groundtruth/image_02/0000000116.png 721.5377'

    data_path = '/media/data3/pxrdata/KITTI/annotation/val/'
    filenames_file = '/home/pxr/pxrProject/DepthEstimation/ConvGuidedDepth/data_splits/eigen_train_files_with_gt.txt'
    filenames_file_eval = '/home/pxr/pxrProject/DepthEstimation/ConvGuidedDepth/data_splits/eigen_test_files_with_gt.txt'

    base = '/media/data3/pxrdata/KITTI/image/'
    time_stemp = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

    match_depth(filenames_file_eval, data_path)






