import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import seaborn as sns
import itertools
import csv
from sklearn.metrics import roc_auc_score

MIN_LINSPACE = 8
# seaborn defaults
# sns.set()

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Consolas"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"


# Arg Parser
parser = argparse.ArgumentParser()
parser.add_argument('--ins', type=str, default="plot_ins/ins.yaml",
                         help="Instructions for plotting")
parser.add_argument('--std', default=False, action='store_true', help='Plot standard deviation')                        
args = parser.parse_args()

# Import plot settings
with open(args.ins, 'r') as file:
    ins = yaml.safe_load(file)

num_x = ins['num_tasks']
y_name=ins["ys"]


# Import results
for name, results_header in ins["results"].items(): 

    # save name
    outdir = "plots_and_tables/" + ins["save_dir"] + '/' + ins["save_name"] + '/' + results_header["file"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = outdir + '/'
    for metric_prefix in ['acc','aux_task']:
        results_all = {}
        res_dir = ins["dir"] + '/' + results_header["file"]
        file_search_top =  [f.path for f in os.scandir(res_dir) if f.is_dir()]
        for file_top in file_search_top:
            file_search = [f.path for f in os.scandir(file_top) if f.is_dir()]
            for file_sweep in file_search:
                try:
                    with open(file_sweep+'/results-' + metric_prefix + '/global.yaml', 'r') as file:
                        results_all[os.path.basename(file_top)+'-'+os.path.basename(file_sweep)] = yaml.safe_load(file)
                except:
                    print('Could not load: ' + file_sweep+'/results-' + metric_prefix)

        # get x, y, and legend names from results
        # wait to plot until parsed all results - want
        # to plot in order of decreasing final accuracy for easy
        # interpretation
        #
        x_plot = []
        y_plot = []
        y_plot_history = []
        std_plot = []
        name_plot = []
        final_acc = []
        init_acc = []
        offline = []

        for name, results in results_all.items():
            
            y = np.asarray(results['mean'])
            y_history = np.asarray(results['history'])
            num_x = ins['num_tasks']
            if len(y) == 1: 
                x = ins['split_size'] * (np.arange(num_x)+1)
                y = y * np.ones((num_x,))
                offline.append(True)
            else:
                x = ins['split_size'] * (np.arange(num_x)+1)
                offline.append(False)
            x_plot.append(x)
            y_plot.append(y)
            y_plot_history.append(y_history)
            try:
                std_plot.append(results['std'])
            except:
                std_plot.append(np.std(np.asarray(y_history),axis=1))
            
            name_plot.append(name)
            final_acc.append(y[-1])
            init_acc.append(y[0])

        final_acc = np.asarray(final_acc)
        csv_rows = []
        row_head = ['Learner']
        row_head.extend(['Mean'])
        row_head.extend(['Std'])
        csv_rows.append(row_head)
        for i in range(len(name_plot)):
            j = np.argsort(final_acc)[-i-1]
            print(name_plot[j])
            results_row = [name_plot[j]]
            #results_row.extend(['{acc:.3f}'.format(acc=acc) for acc in y_plot[j]])
            results_row.extend(['{acc:.1f}'.format(acc=y_plot[j][-1])])
            results_row.extend(['{acc:.1f}'.format(acc=std_plot[j][-1])])
            csv_rows.append(results_row)

        with open(outfile+metric_prefix+'.csv', mode='w') as save_file:
            csv_writer = csv.writer(save_file, delimiter=',')
            for t in range(len(csv_rows)): csv_writer.writerow(csv_rows[t])