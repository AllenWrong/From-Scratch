import utils
import os
import pandas as pd


log_root = './metric_logs'
logs = os.listdir(log_root)

names = [it[:it.index('_')] for it in logs]

log_list = []
for log in logs:
    log_list.append(utils.load_list(os.path.join(log_root, log)))

df = pd.DataFrame(log_list).T
df.columns = names

utils.plot(df, 'imgs/dice_curves.png')