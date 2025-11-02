import numpy as np


import matplotlib.pyplot as plt

'''
---------------------------
NOT FINISHED; DOESN'T WORK
---------------------------
I tried to plot the RA, DE and radial velocity from table 5,
but due to the format of the txt file i couldn't figure out how to 
correctly assign the values to each star.
The problem is that not every cell contains values. E.g. at t=2001.502 has
no data for S1 but it does for S2
'''

filename =  "table5.dat.txt"
#full_data = np.genfromtxt(filename, usecols=range(2, 22), dtype=None)

astrometric_lines = []
radialvel_lines = []
with open(filename, 'r') as f:
    for line in f:
        if 'a' in line:
            astrometric_lines.append(line)
        if 'rv' in line:
            radialvel_lines.append(line)

data =[[]*48] * 168
for i in range(0,168):
    data[i] = "".join(astrometric_lines[i])
    data[i] = data[i].split()
    print(data[i])


for i in range(2,48, 6):
    for val in range(len(data)):

        x = float(data[val][i])
        y = float(data[val][i+2])
        print(x)
        print(y)
    plt.plot(x,y)


plt.scatter(0, 0, color='black', marker='+', label='Sgr A*')
plt.xlabel('R.A. ["]')
plt.ylabel('Dec. ["]')
plt.tight_layout()
plt.gca().invert_xaxis()
plt.grid(True)
#plt.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=len(names_val), fontsize=7)
plt.show()


