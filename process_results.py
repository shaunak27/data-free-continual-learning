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

MIN_LINSPACE = 9
# seaborn defaults
# sns.set()

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Consolas"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"

# incremental learning metric
def calc_omega(y, oracle_y, index=None):
    y = np.asarray(y)
    oracle_y = np.asarray(oracle_y)
    y = y.T
    oracle_y = oracle_y.T
    omega=[]
    if index is None:
        index = y.shape[1]
    for r in range(min(len(y),len(oracle_y))):
        omega_r = 0
        for t in range(index):
            omega_r+=y[r][t]/oracle_y[r][t]
        omega.append(omega_r/index)

    omega = np.asarray(omega)*100
    return np.mean(omega),np.std(omega)

# incremental forgetting metric
# ASSUMES EQUAL TASK SIZE
def calc_forgetting(y, index=None):
    y = np.asarray(y)
    y = y.T
    fgt_all=[]
    if index is None:
        index = y[0].shape[1]
    for r in range(len(y)):
        fgt=0
        for t in range(1,index):
            for i in range(t):
                fgt+= (y[r][i,i] - y[r][t,i]) * ((1)/(t+1))
        fgt_all.append(fgt / (len(y[r])-1))
    fgt_all = np.asarray(fgt_all)
    return np.mean(fgt_all), np.std(fgt_all)

# incremental BWT metric
# ASSUMES EQUAL TASK SIZE
def calc_BWT(y, index=None):
    y = np.asarray(y)
    y = y.T
    fgt_all=[]
    if index is None:
        index = y[0].shape[1]-1
    for r in range(len(y)):
        fgt=0
        for t in range(0,index):
            fgt+= (y[r][-1,t] - y[r][t,t])
        fgt_all.append(fgt / (len(y[r])-1))
    fgt_all = np.asarray(fgt_all)
    return np.mean(fgt_all), np.std(fgt_all)


# Arg Parser
parser = argparse.ArgumentParser()
parser.add_argument('--ins', type=str, default="plot_ins/ins.yaml",
                         help="Instructions for plotting")
parser.add_argument('--std', default=False, action='store_true', help='Plot standard deviation')                        
args = parser.parse_args()

# Import plot settings
with open(args.ins, 'r') as file:
    ins = yaml.safe_load(file)

# save name
outdir = "plots_and_tables/" + ins["save_dir"] + '/' + ins["save_name"]
if not os.path.exists(outdir):
    os.makedirs(outdir)
outfile = outdir + '/'

num_x = ins['num_tasks']
y_name=ins["ys"]

for same_plot in [True, False]:
    same_plot_idx = 0
    rows = 2
    cols = 3
    sample_plot_title = outfile+'_main_results'
    if same_plot: plt.figure(figsize=(24,12))

    for metric_prefix in ['acc','mem','time','plastic','til','cka']:

        acc_flag = metric_prefix == 'acc'

        # Import results
        results_all = {}
        results_all_pt = {}
        results_all_pt_b = {}
        c_index_map = {}
        for name, results_header in ins["results"].items(): 
            c_index_map[results_header["name"]] = int(name)
            if int(name) >= 0:
                try:
                    with open(results_header["file"]+'/results-' + metric_prefix + '/global.yaml', 'r') as file:
                        results_all[results_header["name"]] = yaml.safe_load(file)
                    if int(name) > 0 and acc_flag:
                        with open(results_header["file"]+'/results-' + metric_prefix + '/pt.yaml', 'r') as file:
                            results_all_pt[results_header["name"]] = yaml.safe_load(file)
                        with open(results_header["file"]+'/results-' + metric_prefix + '/pt-local.yaml', 'r') as file:
                            results_all_pt_b[results_header["name"]] = yaml.safe_load(file)
                except:
                    print('Could not load '+str(results_header))
            else:
                print('Ignoring '+str(results_header))

        metrics = ['Accuracy', 'Omega', 'Task FGT', 'Class FGT']
        for mindex in range(len(metrics)):

            metric = metrics[mindex]
            if metric == 'Accuracy' or acc_flag:

                # get x, y, and legend names from results
                # wait to plot until parsed all results - want
                # to plot in order of decreasing final accuracy for easy
                # interpretation
                #
                x_plot = []
                y_plot = []
                y_plot_pt = []
                y_plot_pt_b = []
                y_plot_history = []
                std_plot = []
                name_plot = []
                final_acc = []
                init_acc = []
                offline = []

                oracle_flag = None
                for name, results in results_all.items():
                    
                    if name == 'Oracle':
                        oracle_flag = True
                        oracle_y = np.asarray(results['mean'])
                        oracle_y_history = np.asarray(results['history'])
                    if not name == 'Oracle' and acc_flag:
                        y_pt = np.asarray(results_all_pt[name]['history'])
                        y_pt_b = np.asarray(results_all_pt_b[name]['history'])


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
                        if not name == 'Oracle' and acc_flag: 
                            y_plot_pt.append(y_pt)
                            y_plot_pt_b.append(y_pt_b)
                    if not name == 'Oracle':
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

                if not oracle_flag and acc_flag:
                    raise ValueError('No oracle found!!!')

                csv_rows = []
                row_head = ['Learner']
                # row_head.extend(['T-' + str(t+1) + ' Acc' for t in range(num_x)])
                if oracle_flag:
                    row_head.extend(['An-Mean'])
                    row_head.extend(['An-Std'])
                    row_head.extend(['Omega-Mean'])
                    row_head.extend(['Omega-Std'])
                row_head.extend(['Task FGT-Mean'])
                row_head.extend(['Task FGT-Std'])
                row_head.extend(['Class FGT-Mean'])
                row_head.extend(['Class FGT-Std'])
                csv_rows.append(row_head)

                if acc_flag:
                    for j in range(len(name_plot)):
                        print(name_plot[j])
                        results_row = [name_plot[j]]
                        #results_row.extend(['{acc:.3f}'.format(acc=acc) for acc in y_plot[j]])
                        results_row.extend(['{acc:.1f}'.format(acc=y_plot[j][-1])])
                        results_row.extend(['{acc:.1f}'.format(acc=std_plot[j][-1])])
                        if oracle_flag: 
                            omega = calc_omega(y_plot_history[j], oracle_y_history)
                            results_row.extend(['{acc:.1f}'.format(acc=omega[0])])
                            results_row.extend(['{acc:.1f}'.format(acc=omega[1])])
                        if name_plot[j] == 'Oracle':
                            results_row.extend(['{acc:.1f}'.format(acc=0.0)])
                            results_row.extend(['{acc:.1f}'.format(acc=0.0)])
                            results_row.extend(['{acc:.1f}'.format(acc=0.0)])
                            results_row.extend(['{acc:.1f}'.format(acc=0.0)])
                        else:
                            bwt = calc_BWT(y_plot_pt_b[j])
                            results_row.extend(['{acc:.1f}'.format(acc=bwt[0])])
                            results_row.extend(['{acc:.1f}'.format(acc=bwt[1])])
                            fgt = calc_forgetting(y_plot_pt[j])
                            results_row.extend(['{acc:.1f}'.format(acc=fgt[0])])
                            results_row.extend(['{acc:.1f}'.format(acc=fgt[1])])
                        csv_rows.append(results_row)

                if not metric == 'Accuracy' and acc_flag:
                    final_acc = []
                    init_acc = []
                    y_plot_new = []
                    std_plot_new = []
                    for i in range(len(y_plot)):
                        y = y_plot[i]
                        y_metric = []
                        y_std = []
                        if metric == 'Omega':
                            y_metric.append(100)
                        elif metric == 'Task FGT':
                            y_metric.append(0)
                        elif metric == 'Class FGT':
                            y_metric.append(0)
                        y_std.append(0.0)
                        for t in range(1,num_x):
                            if metric == 'Omega':
                                result = calc_omega(y_plot_history[i], oracle_y_history, index = t+1)
                                y_metric.append(result[0])
                                y_std.append(result[1])
                            elif metric == 'Task FGT':
                                result = calc_BWT(y_plot_pt_b[i], index = t)
                                y_metric.append(-1 * result[0])
                                y_std.append(result[1])
                            elif metric == 'Class FGT':
                                result = calc_forgetting(y_plot_pt[i], index = t+1)
                                y_metric.append(result[0])
                                y_std.append(result[1])
                        y_plot_new.append(np.asarray(y_metric))
                        std_plot_new.append(np.asarray(y_std))
                        final_acc.append(y_metric[-1])
                        init_acc.append(y_metric[0])
                    y_plot = y_plot_new
                    std_plot = std_plot_new  

                # array for sorting
                final_acc = np.asarray(final_acc)
                init_acc = np.asarray(init_acc)
                # plot with matplotlib
                cmap = plt.get_cmap('jet')
                marks = ['p', 'D', 'X', 'o', '*','v']
                lines = ['solid','dashed']
                max_linspace = max(MIN_LINSPACE, len(c_index_map.items()))
                colors = cmap(np.linspace(0, 1.0, max_linspace))
                if same_plot:
                    same_plot_idx += 1
                    if same_plot_idx > 6:
                        same_plot_idx = 1
                        rows = 1
                        sample_plot_title = outfile+'_side_results'
                        plt.figure(figsize=(24,6))
                    plt.subplot(rows, cols, same_plot_idx)
                else:
                    plt.figure(figsize=(8,4))
                for i in range(len(name_plot)):

                    # get next highest accuracy
                    j = np.argsort(final_acc)[-i-1]
                    c_index = max_linspace - c_index_map[name_plot[j]]

                    # if offline
                    if offline[j]:

                        # add to plot
                        plt.plot(x_plot[j], y_plot[j], color = colors[c_index], label = name_plot[j], linestyle = lines[c_index % len(lines)], lw=2)
                        
                        # add standard deviation
                        if args.std:
                            sigma = np.asarray(std_plot[j])
                            plt.fill_between(x_plot[j], y_plot[j]+sigma, y_plot[j]-sigma, facecolor=colors[c_index], alpha=0.3)

                    # else
                    else:

                        # add to plot
                        plt.plot(x_plot[j], y_plot[j], color = colors[c_index], lw=2, linestyle = lines[c_index % len(lines)])
                        plt.scatter(x_plot[j], y_plot[j], color = colors[c_index], label = name_plot[j], marker = marks[c_index % len(marks)],s=50)
                        
                        # add standard deviation
                        if args.std:
                            sigma = np.asarray(std_plot[j])
                            plt.fill_between(x_plot[j], y_plot[j]+sigma, y_plot[j]-sigma, facecolor=colors[c_index], alpha=0.3)

                            
                # axis and stuff
                if acc_flag:
                    plt.yticks(np.arange(-10, 110, 10),fontsize=14)
                    miny=min(20,np.floor(np.amin(np.asarray(y_plot))/10)*10)
                    maxy=max(101,np.ceil(np.amax(np.asarray(y_plot))/10)*10)
                    plt.ylim(int(miny),int(maxy))
                    # plt.ylim(np.floor(np.amin(np.asarray(y_plot))/10)*10, np.ceil(np.amax(np.asarray(y_plot))/10)*10)
                    
                if metric == 'Omega':
                    plt.ylabel(r'$\Omega (\%)$', fontweight='bold', fontsize=18)
                elif metric_prefix == 'mem':
                    plt.ylabel(' # Stored Parameters', fontweight='bold', fontsize=18)
                elif metric_prefix == 'time':
                    plt.ylabel('Time (sec per batch)', fontweight='bold', fontsize=18)
                elif metric_prefix == 'plastic':
                    plt.ylabel('Task N Accuracy (%)', fontweight='bold', fontsize=18)
                elif metric_prefix == 'til':
                    plt.ylabel('Task Incremental Accuracy (%)', fontweight='bold', fontsize=18)
                elif metric_prefix == 'cka':
                    plt.ylabel('CKA', fontweight='bold', fontsize=18)
                else:
                    plt.ylabel(metric + ' (%)', fontweight='bold', fontsize=18)
                plt.xlabel(ins["xs"], fontweight='bold', fontsize=18)
                tick_x = ins["split_size"] * np.arange(1+ins['num_tasks'])
                tick_x_s = []
                if len(tick_x) <= 10:
                    for tick in tick_x:
                        if tick % ins["split_size"] == 0:
                            tick_x_s.append(str(tick))
                        else:
                            tick_x_s.append('')
                else:
                    for tick in tick_x:
                        if tick % (ins["split_size"] * 2) == 0:
                            tick_x_s.append(str(tick))
                        else:
                            tick_x_s.append('')
                plt.xticks(tick_x, tick_x_s,fontsize=14)
                if metric_prefix == 'cka':
                    plt.xlim(2)
                else:
                    plt.xlim(ins["split_size"]*.9)
                # plt.title(ins["title"], fontweight='bold', fontsize=14)
                
                if len(name_plot) > 10:
                    legend_font = 6
                elif len(name_plot) > 8:
                    legend_font = 7
                elif len(name_plot) > 6:
                    legend_font = 8
                else:
                    legend_font = 10
                if metric_prefix == 'time':
                    plt.legend(loc='upper left', prop={'weight': 'bold', 'size': legend_font})
                elif metric == 'Task FGT' or metric == 'Class FGT':
                    plt.legend(loc='upper left', prop={'weight': 'bold', 'size': legend_font})
                elif not acc_flag:
                    plt.legend(loc='lower right', prop={'weight': 'bold', 'size': legend_font})
                else:
                    plt.legend(loc='lower left', prop={'weight': 'bold', 'size': legend_font})
                plt.grid()
                
                if same_plot:
                    plt.savefig(sample_plot_title+'.png') 
                    print(outfile)  
                else:
                    plt.tight_layout()
                    if acc_flag:
                        plt.savefig(outfile+metric+'.png') 
                        print(outfile)    
                        plt.close()
                    else:
                        plt.savefig(outfile+metric_prefix+'.png') 
                        print(outfile)    
                        plt.close()

        # if not acc_flag:
        if acc_flag:
            with open(outfile+'metrics.csv', mode='w') as save_file:
                csv_writer = csv.writer(save_file, delimiter=',')
                for t in range(len(csv_rows)): csv_writer.writerow(csv_rows[t])

            # write as latex table rows
            with open(outfile+'LATEX.txt', mode='w') as save_file:
                for item in csv_rows[0]:
                    save_file.write(str(item))
                    save_file.write('   ')
                save_file.write('\n')
                for t in range(1,len(csv_rows)):
                    save_file.write(str(csv_rows[t][0]))
                    save_file.write(' ')
                    for j in range(2):
                        save_file.write(' & ')
                        save_file.write('$ ')
                        save_file.write(csv_rows[t][j*2 + 1])
                        # save_file.write(' \pm 0.0')
                        save_file.write(' \pm ')
                        save_file.write(csv_rows[t][j*2 + 2])
                        save_file.write(' $')
                    save_file.write('  \\\\ ')
                    save_file.write('\n')