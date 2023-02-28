import glob
import sys
import random

import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from gprMax import gprMax
from tools.outputfiles_merge import get_output_data
from tools.outputfiles_merge import merge_files



def generate_in_files(filename: str, type: str = 'all', seed: int = 1000) -> bool:
    """
    生成输入文件
    :param type: all:既有空洞又有目标  obj:只有目标  none:啥都没有
    :return:
    """
    file = open(f'{filename}_{type}.in', 'w')

    random.seed(seed)
    cent_circle_x = random.uniform(0.3, 1.2)
    cent_circle_x = int(cent_circle_x * 1000) / 1000

    random.seed(seed)
    cent_circle_y = random.uniform(0.55, 0.7)
    cent_circle_y = int(cent_circle_y * 1000) / 1000

    random.seed(seed)
    radius_1 = random.uniform(0.03, 0.045)
    radius_1 = int(radius_1 * 1000) / 1000

    random.seed(seed)
    radius_2 = random.uniform(0.015, 0.02)
    radius_2 = int(radius_2 * 1000) / 1000

    random.seed(seed)
    cent_circle_x1 = random.uniform(0.2, 1.3)
    cent_circle_x1 = int(cent_circle_x1 * 1000) / 1000

    random.seed(seed)
    cent_circle_y1 = random.uniform(0.1, 0.35)
    cent_circle_y1 = int(cent_circle_y1 * 1000) / 1000

    x1, y1, r1 = cent_circle_x, cent_circle_y, radius_1
    x2, y2, r2 = cent_circle_x1 , cent_circle_y1, radius_2

    file.write(
        '#domain: 1.500 0.900 0.002\n'
        '#dx_dy_dz: 0.002 0.002 0.002\n'
        '#time_window: 20e-9\n'
        '#material: 6 0 1 0 half_space\n'
        '#material: 81 0.05 1 0 water\n'
        '#material: 4 0.004 1 0 layer2\n'
        '#material: 9 0.005 1 0 layer3\n'
        '#material: 12 0.003 1 0 layer4\n'
        '#material: 3.5 0 1 0 pvc\n'
        '#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.05 my_soil\n'

        '#waveform: ricker 1 800e6 my_ricker\n'
        '#hertzian_dipole: z 0.040 0.800 0 my_ricker\n'
        '#rx: 0.045 0.800 0\n'
        '#src_steps: 0.02 0 0\n'
        '#rx_steps: 0.02 0 0\n'

        '#box: 0 0.8 0 1.5 0.9 0.002 free_space\n'
        f'#fractal_box: 0 0 0 1.5 0.8 0.002 1.5 5 4 1 50 my_soil mixlayer4 {seed}\n'
        'box: 0 0.7 0 1.5 0.8 0.002 layer2\n'
        'box: 0 0.60 0 1.5 0.70 0.002 layer3\n'
        'box: 0 0 0 1.5 0.60 0.002 layer4\n'
    )
    if type == 'all' or type == 'water':
        file.write(
            f'#cylinder: {x1} {y1} 0 {x1} {y1} 0.002 {r1} pvc\n'
            f'#cylinder: {x1} {y1} 0 {x1} {y1} 0.002 {r1 - 0.01} water\n'
        )
    if type == 'all' or type == 'obj':
        file.write(
            f'#cylinder: {x2} {y2} 0 {x2} {y2} 0.002 {r2} pec\n'
            f'#cylinder: {x2} {y2} 0 {x2} {y2} 0.002 {r2 - 0.01} free_space\n'
        )
    file.close()
    return True




def eliminate_background(file1, file2, output_file='eilm_bg.out'):
    """对Bscan图的.out文件进行背景对消,然后生成一个背景对消后的.out文件
    参数：
        file1：有要探测目标的文件名
        file2: 无要探测目标的文件名
        output_file: 消除背景后的文件名，默认为"elim_bg.out"
    返回值：
        返回一个消除了背景的.out文件
    """
    fin1 = h5py.File(file1 + '.out', 'r')
    fin2 = h5py.File(file2 + '.out', 'r')
    fout = h5py.File(output_file, 'w')

    # 设置输出文件的一些参数
    fout.attrs['Title'] = fin1.attrs['Title']
    fout.attrs['gprMax'] = fin1.attrs['gprMax']
    fout.attrs['Iterations'] = fin1.attrs['Iterations']
    fout.attrs['dt'] = fin1.attrs['dt']
    fout.attrs['nrx'] = 1
    col = fin1['rxs/rx1/Ez'].shape[1]
    for rx in range(1, 2):
        path = '/rxs/rx' + str(rx)
        grp = fout.create_group(path)
        availableoutputs = list(fin1[path].keys())
        for output in availableoutputs:
            grp.create_dataset(output, (fout.attrs['Iterations'], col), dtype=fin1[path + '/' + output].dtype)

    # 进行背景对消
    path = 'rxs/rx1/'

    output = 'Ez'
    for i in range(fout.attrs['Iterations']):
        fout[path + output][i:] = fin1[path + output][i:] - fin2[path + output][i:]

    # 关闭文件
    fin1.close()
    fin2.close()
    fout.close()


def save_bscan_img(filename: str, pos: str, dir_path: str = './img') -> bool:
    from skimage import transform
    rxnumber = 1
    rxcompoent = 'Ez'
    outputdata, dt = get_output_data(f'{filename}.out', rxnumber, rxcompoent)
    outputdata = transform.resize(outputdata, (256, 256))
    plt.imshow(outputdata, cmap=matplotlib.colormaps['gray'])
    plt.imsave(f'{dir_path}/{pos}.jpg', outputdata, cmap=matplotlib.colormaps['gray'])

def start(basename: str, n: int, begin: int, end: int) -> bool:
    for i in range(begin, end):
        seed = random.randint(0, 10000)
        generate_in_files(f'./in/{basename}{i}', 'all', seed)
        generate_in_files(f'./in/{basename}{i}', 'water', seed)
        generate_in_files(f'./in/{basename}{i}', 'obj', seed)
        generate_in_files(f'./in/{basename}{i}', 'none', seed)
        gprMax.api(f'./in/{basename}{i}_all.in', n, gpu=[0])
        merge_files(f'./in/{basename}{i}_all', removefiles=True)
        gprMax.api(f'./in/{basename}{i}_water.in', n, gpu=[0])
        merge_files(f'./in/{basename}{i}_water', removefiles=True)
        gprMax.api(f'./in/{basename}{i}_obj.in', n, gpu=[0])
        merge_files(f'./in/{basename}{i}_obj', removefiles=True)
        gprMax.api(f'./in/{basename}{i}_none.in', n, gpu=[0])
        merge_files(f'./in/{basename}{i}_none', removefiles=True)
        eliminate_background(f'./in/{basename}{i}_all_merged', f'./in/{basename}{i}_none_merged',
                             f'./out/all_eilm_bg{i}.out')
        eliminate_background(f'./in/{basename}{i}_obj_merged', f'./in/{basename}{i}_none_merged',
                             f'./out/obj_eilm_bg{i}.out')
        eliminate_background(f'./in/{basename}{i}_water_merged', f'./in/{basename}{i}_none_merged',
                             f'./out/water_eilm_bg{i}.out')

        save_bscan_img(f'./out/all_eilm_bg{i}', f'x_all_{i}', './imgs')
        save_bscan_img(f'./out/water_eilm_bg{i}', f'water_{i}', './imgs')
        save_bscan_img(f'./out/obj_eilm_bg{i}', f'obj_{i}', './imgs')

if __name__ == '__main__':
    start("third", 72, 10, 11)