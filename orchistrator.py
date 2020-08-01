import os

os.system('python training.py -i 18 -o 8 --decomposed')

base = 'python training.py -i 18 -o 8 -a {} --decomposed'
augs = [0.5, 0.67, 0.75, 0.8, 0.9, 0.95]

for a in augs:
    os.system(base.format(str(a)))

