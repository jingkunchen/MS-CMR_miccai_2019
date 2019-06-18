from __future__ import print_function
from __future__ import division

import click
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def visualize_test_evaluation(input_file, output_folder, experiment_title = ''):
    assert os.path.isfile(input_file)

    with open(input_file) as infile:
        eval_dict = json.load(infile)

    dsc_list = eval_dict['dsc']
    h95_list = eval_dict['h95']
    vs_list = eval_dict['vs']

    dsc_avg_list = [np.nanmean(np.array(item.values(), dtype='float32')) for item in dsc_list]
    h95_avg_list = [np.nanmean(np.array(item.values(), dtype='float32')) for item in h95_list]
    vs_avg_list = [np.nanmean(np.array(item.values(), dtype='float32')) for item in vs_list]

    x_axis = list(range(0, len(dsc_avg_list)*10, 10))


    plt.plot(x_axis, dsc_avg_list, 'r', x_axis, vs_avg_list, 'g')
    plt.title('Average Dice Coefficient and Volume similarity \n' + experiment_title)
    plt.legend(['Avg. Dice Coefficient', 'Avg. Volume Similarity'])
    plt.text(0.4 * len(dsc_avg_list) * 10, 0.80, 'max. dice coefficient: ' + str(np.max(dsc_avg_list)))
    plt.text(0.4 * len(dsc_avg_list) * 10, 0.75, 'max. volume similarity: ' + str(np.max(vs_avg_list)))
    plt.savefig(os.path.join(output_folder, 'dice_vs_graph.png'))
    plt.close()

    plt.plot(x_axis, h95_avg_list, 'b')
    plt.title('Average Hausdorff distance \n' + experiment_title)
    plt.legend(['Avg. Housdorff distance'])
    plt.savefig(os.path.join(output_folder, 'h95.png'))
    plt.close()

    return 0

@click.command()
@click.argument('input_file', type=click.STRING)
@click.argument('output_folder', type=click.STRING)
@click.option('--experiment_title', type=click.STRING, default='')
def main(input_file, output_folder, experiment_title=''):
    visualize_test_evaluation(input_file, output_folder, experiment_title=experiment_title)
    return 0

if __name__ == '__main__':
    main()