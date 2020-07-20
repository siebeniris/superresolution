import json
import os

import matplotlib.pyplot as plt
from statistics import mean


boxes_5m_file='data/interim/coords_train_dict.json'
boxes_30m_file = 'data/interim/slices_train_dict.json'


boxes_5m = json.load(open(boxes_5m_file))
boxes_30m = json.load(open(boxes_30m_file))


def get_statistics(boxes_list):
    areas=[]
    heights=[]
    lengths=[]
    for boxes in boxes_list:
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            length= xmax-xmin
            height= ymax -ymin
            areas.append(length*height)
            heights.append(height)
            lengths.append(length)

    area_mean, height_mean, length_mean= mean(areas), mean(heights), mean(lengths)
    area_max, height_max, length_max= max(areas), max(heights), max(lengths)
    area_min, height_min, length_min = min(areas), min(heights),min(lengths)
    print('----- mean, min, max')
    print('area: {}, {}, {}'.format(area_mean, area_min,area_max))
    print('height: {}, {}, {}'.format(height_mean, height_min, height_max))
    print('length: {}, {}, {}'.format(length_mean, length_min, length_max))
    return areas, heights, lengths


def plot(ls, xlabel, filename):
    plt.hist(ls, bins='auto',range=(0, 148))
    plt.title('Histogram of 30m images length distribution ')
    plt.ylabel('Amount of seedlings')
    plt.xlabel(xlabel)
    plt.savefig(filename)
    plt.show()





if __name__ == '__main__':
    data_dir='data/interim/plots_405/'
    os.mkdir(data_dir)
    print('******** statistics of 30m images *********')

    areas, height, lengths= get_statistics(boxes_30m.values())


    print('******** statistics of 5m images *********')

    get_statistics(boxes_5m.values())

    # plot(lengths, 'Length in Pixel', data_dir+'30m_length.png')
