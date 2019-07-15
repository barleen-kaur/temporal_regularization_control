#This code checks if all of the hyper param configs have run to completion. If there are less than 2,000,000 frames completed for some config, it means it did not finish executing. When you run this code, if there is no output, that means all of the configs ran till completion.

import os

os.chdir(os.path.expanduser("~"))
path = "/scratch/ravitej3/"

min_iters = 10000000000

for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".log")):
            with open(os.path.join(root,name), 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                #print(last_line)
                substr = last_line[last_line.find(",")+1:last_line.find(".0,")]
                substr = int(substr)
                #print(substr)
                if substr<2000000:
                    print(os.path.join(root,name))
                #if substr<min_iters and substr>160000:
                #    min_iters = substr
                #    path = os.path.join(root,name)


#print(min_iters)
#print(path)
