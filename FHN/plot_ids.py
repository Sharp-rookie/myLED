import matplotlib.pyplot as plt

fp = open('logs/ID.txt', 'r')
tau_list = []
id_list = []
for id_str in fp.readlines():
    tau = id_str[:-1].split('--')[0]
    id = id_str[:-1].split('--')[1]
    tau_list.append(float(tau))
    id_list.append(float(id))
plt.figure()
plt.scatter(tau_list, id_list)
plt.savefig('ids.jpg', dpi=300)