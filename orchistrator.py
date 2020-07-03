import os

base = 'python training.py -i 18 -o 8 '
augs = [0.5, 0.67, 0.75, 0.8, 0.9, 0.95]

for a in augs:
    os.system(base + str(a))
