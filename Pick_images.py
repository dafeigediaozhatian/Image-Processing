import os
import shutil

main_root = '/home/flyingbird/Flyingbird/Test'
image_file = main_root + '/images'
source_path = main_root + '/Sets_list/train_set.txt'

def get_images_name(source_path, save_path):
    """
    input: 原路径列表，保存路径列表（str)
    output: 是否执行成功(yes or no)
    """
    with open(source_path, 'r') as f:
        images_list = []
        for i in f.readlines():
            images_list.append(i.strip().split(' ')[1] + '.jpg')
        images_list = images_list[1:]
    
    for images_name in images_list:
        images_id = os.path.join(image_file, images_name)
        shutil.copy(images_id, save_path)
    
    if len(os.listdir(save_path)) == 5000:
        print('Copy Success')
