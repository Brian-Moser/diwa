import json
import os

def create_data_list(folders, output_folder, ds_name):
    print("\nCreating data list <" + ds_name + ">...")
    images = list()
    for d in folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            images.append(img_path)
    print("Total amount: %d images" % len(images))
    with open(os.path.join(output_folder, '' + ds_name + '_images.json'), 'w') as j:
        json.dump(images, j)

if __name__ == '__main__':
    create_data_list(folders=['./dataset/Set5/original/'],
                     output_folder='./dataset/',
                     ds_name='set5'
                     )
    create_data_list(folders=['./dataset/Set14/original/'],
                     output_folder='./dataset/',
                     ds_name='set14'
                     )
    create_data_list(folders=['./dataset/BSDS100'],
                     output_folder='./dataset/',
                     ds_name='bsds100'
                     )
    create_data_list(folders=['./dataset/urban100'],
                     output_folder='./dataset/',
                     ds_name='urban100'
                     )
    create_data_list(folders=['./dataset/General100'],
                     output_folder='./dataset/',
                     ds_name='General100'
                     )
    create_data_list(folders=['./dataset/DIV2K_valid_HR'],
                     output_folder='./dataset/',
                     ds_name='div2k_valid'
                     )