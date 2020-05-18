import os
from pathlib import Path
import sys
from tqdm import tqdm

check = False

if len(sys.argv) > 1:
    if sys.argv[1] == '--check':
        check = True

remote_dir = 'projects/ae_pred/results'
local_dir = 'results'

os.system('ssh fsu-gpu "ls {}" > /tmp/server_results.log'.format(remote_dir))

with open('/tmp/server_results.log') as f:
    remote = [line.rstrip('\n') for line in f]

local = [str(x) for x in Path(local_dir).glob('*')]

to_be_uploaded = list(set(local).difference(remote))

for f in tqdm(to_be_uploaded):
    if check:
        print('scp -r {} fsu-gpu:{}'.format(f, remote_dir))
    else:
        os.system('scp -r {} fsu-gpu:{}'.format(f, remote_dir))
