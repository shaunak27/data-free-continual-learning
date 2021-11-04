import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import seaborn as sns
import itertools
import csv
# seaborn defaults
# sns.set()
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Consolas"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"

base=10
MIN_LINSPACE = 8

# files to load
save_dir = 'ICCV/CIFAR100/extra'
save_name_all = ['memory_study']
file_sweep_all = [['outputs/ICCV/coreset-free-replay-tentask/CIFAR100/base']]
# name_sweep_title = ['extbf{Naive Rehearsal}']
name_sweep_title = ['Naive Rehearsal']
coreset_sizes = [0,100,200,500,1000,2000,5000]
file_straight_all = [['outputs/ICCV/coreset-free-replay-tentask/CIFAR100/ours','outputs/ICCV/coreset-free-replay-tentask/CIFAR100/deepinv']]
# name_straight = ['extbf{Ours}','extbf{Adaptive DeepInv}']
name_straight = ['Ours','DeepInv']
file_oracle_all = ['outputs/ICCV/coreset-free-replay-tentask/CIFAR100/Oracle']

# incremental learning metric
def calc_omega(y, oracle_y):
    y = np.asarray(y['history'])
    oracle_y = np.asarray(oracle_y['history'])
    y = np.asarray(y)
    oracle_y = np.asarray(oracle_y)
    y = y.T
    oracle_y = oracle_y.T
    omega=0
    for r in range(len(y)):
        omega_r = 0
        for t in range(len(y[r])):
            omega_r+=y[r][t]/oracle_y[r][t]
        omega+=omega_r/len(y[r])

    return min(omega/len(y) * 100,100)

for r in range(len(file_sweep_all)):

    # files to load
    save_name = save_name_all[r]
    file_sweep = file_sweep_all[r]
    file_straight = file_straight_all[r]
    file_oracle = file_oracle_all[r]

    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return False, 0

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return True, x

    # load omega
    with open(file_oracle+'/results-acc/global.yaml', 'r') as file:
        loaded_oracle =  yaml.safe_load(file)

    # load straight
    loaded_straight = []
    for straight in file_straight:
        with open(straight+'/results-acc/global.yaml', 'r') as file:
            loaded_straight.append(yaml.safe_load(file))

    # load sweeep
    loaded_sweep = []
    for i in range(len(file_sweep)):
        sweep = file_sweep[i]
        loaded_coreset = []
        for coreset in coreset_sizes:
            with open(sweep+'_'+str(coreset)+'/results-acc/global.yaml', 'r') as file:
                loaded_coreset.append(yaml.safe_load(file))
        loaded_sweep.append(loaded_coreset)

    # calculate omega
    omega_straight = []
    for loaded in loaded_straight:
        omega_straight.append(calc_omega(loaded, loaded_oracle))
    omega_sweep = []
    for loaded in loaded_sweep:
        omega_coreset = []
        for coreset in loaded:
            omega_coreset.append(calc_omega(coreset, loaded_oracle))
        omega_sweep.append(omega_coreset)

    # save name
    outdir = "plots_and_tables/" + save_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = outdir + '/' + save_name

    # plot with matplotlib
    cmap = plt.get_cmap('jet')
    marks = [',', '+', '.', 'o', '*']
    max_linspace = len(omega_sweep) + len(omega_straight)
    colors = cmap(np.linspace(0, 1.0, max_linspace))

    plt.figure(figsize=(8,4))
    # matplotlib.rc('text', usetex=True)
    # matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
    x = np.asarray(coreset_sizes)
    x[x==0] = 50
    x = np.log(x) / np.log(base)
    for i in range(len(omega_sweep)):

        # add to plot
        plt.plot(x, omega_sweep[i], color = colors[i], linestyle = 'solid', lw=3)
        # plt.scatter(x, omega_sweep[i], color = colors[i], label = r'\t{text}'.format(text=name_sweep_title[i]), marker = 'D',s=50)
        plt.scatter(x, omega_sweep[i], color = colors[i], label = name_sweep_title[i], marker = 'D',s=50)

    styles = ['dashed','dotted']
    x = [0,10000]
    x = np.asarray(coreset_sizes)
    x[x==0] = 50
    x = np.log(x) / np.log(base)
    for i in range(len(omega_straight)):

        # add to plot
        # plt.plot(x, omega_straight[i] * np.ones(len(x)), color = colors[i + len(omega_sweep)], label = r'\t{text}'.format(text=name_straight[i]), linestyle = styles[i], lw=2)
        plt.plot(x, omega_straight[i] * np.ones(len(x)), color = colors[i + len(omega_sweep)], label = name_straight[i], linestyle = styles[i], lw=2)

        print('Interpolation for ' + str(name_straight[i]))
        found = False
        j = 0
        x_loop = np.asarray(coreset_sizes)
        while not found:
            linea = [[0,omega_straight[i]],[10000,omega_straight[i]]]
            lineb = [[x_loop[j],omega_sweep[0][j]],[x_loop[j+1],omega_sweep[0][j+1]]]
            found, intersection = line_intersection(linea,lineb)

            if omega_sweep[0][j+1] < omega_straight[i]:
                found = False

            j += 1
        print(intersection)

    # axis and stuff
    plt.yticks(np.arange(10, 110, 10),fontsize=14)
    # plt.xlabel(r'\textbf{Coreset Size (log scaled)}', fontweight='bold', fontsize=18)
    # plt.ylabel(r'$\Omega (\%)$', fontweight='bold', fontsize=18)
    plt.xlabel('Coreset Size (log scaled)', fontweight='bold', fontsize=18)
    plt.ylabel('$\Omega (\%)$', fontweight='bold', fontsize=18)
    x_t_l = np.asarray([0,100,200,500,1000, 2000])
    x_t = np.asarray([0,100,200,500,1000, 2000])
    x_t[x_t==0] = 50
    x_t = np.log(x_t) / np.log(base)
    plt.xlim([x_t[0],x_t[1]])
    # plt.xticks(x_t, [r'${text}$'.format(text=coreset) for coreset in x_t_l],fontsize=14)
    plt.xticks(x_t, [str(coreset) for coreset in x_t_l],fontsize=14)
    plt.ylim(15,65)  
    plt.xlim(x[0])
    plt.grid()
    plt.legend(loc='lower right', prop={'weight': 'bold', 'size': 15})
    plt.tight_layout()
    plt.savefig(outfile+'.png') 
    plt.close()