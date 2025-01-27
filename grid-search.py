
import os
import time, datetime
import argparse
import shutil

parser = argparse.ArgumentParser()

# Arquitetura ['alexnet', 'resnet101', 'densenet121']
parser.add_argument('--ds', type=str, default='CropPestAndDiseaseDetection', help='Dataset name.')
parser.add_argument('--arch', type=str, default='vit', help='Arquitetura da CNN. Default: alexnet.')

parser.add_argument('--optimizer', help="Optimizer. ['SGD', 'Adam'].", type=str, default='Adam')
parser.add_argument('--scheduler', help="Scheduler. ['steplr', 'cossine', 'plateau'].", type=str, default='plateau')

args = parser.parse_args()

# Hyperparameter optmization
args.optim = 'grid'
# Number of epochs
ep = 200 # 50 # 400
# Early stoppping
ES = True

EXP_PATH_MAIN = f'exp_hp_{args.ds}'

# Is debug?
DEBUG = False

if DEBUG:
    bs_list = [16, 32,]
    lr_list = [0.01, 0.001,]
    mm_list = [0.9,]
    ss_list = [5, 10,]

else:
    ### bs_list = [32, 64, 128, 256]
    bs_list = [16, 32, 64, 128]
    lr_list = [0.01, 0.001, 0.0001, 0.00001]
    # https://towardsdatascience.com/why-0-9-towards-better-momentum-strategies-in-deep-learning-827408503650
    mm_list = [0.9, 0.95, 0.99]
    ### ss_list = [5, 10, 20]
    ss_list = [0, 5, 10, 25]

if args.optimizer == 'Adam': mm_list = [0]
if args.scheduler != 'steplr': ss_list = [0]

str_es = '--es' if ES else '--no-es'

# Hyperparameter optmization summary:
# ---
# 4 x 4 x 3 x 3 = 144 experiments
# 4 x 4 = 16 experimentos, se Adam + Plateau

# Experiment counter
ec = 0

# Inicia contagem de tempo deste época
time_start = time.time()

for bs in bs_list:
    for lr in lr_list:
        for mm in mm_list:
            for ss in ss_list:
                
                cmd_str = f'nohup python CNNPestes.py --ds {args.ds} --da 0 --arch {args.arch} ' + \
                          f'--optim {args.optim} --bs {bs} --lr {lr} --mm {mm} --ss {ss} --ep {ep} ' + \
                          f'--optimizer {args.optimizer} --ft --ec {ec} {str_es} --patience 21 ' + \
                          f'--num_workers 0 '

                ec = ec + 1

                os.system(cmd_str)

time_exp = time.time() - time_start
time_exp_hms = str(datetime.timedelta(seconds = time_exp))
print(f'Time exp.: {time_exp} sec ({time_exp_hms})')

if os.path.exists('./nohup.out'):
    shutil.move('./nohup.out', os.path.join(EXP_PATH_MAIN, '(' + args.ds + ')-' + args.arch + '-' + args.optim + '-nohup.out'))

# Cria o arquivo txt
report_file = open('exp_report.txt', 'w')
report_file.write(f'Time exp.: {time_exp} sec ({time_exp_hms})')
report_file.close()

print('Done! (grid_search)')