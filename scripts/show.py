import pickle
from matplotlib import pyplot as plt

with open("g_list.pkl", "rb") as f:
    g_list = pickle.load(f)

with open("loss_list.pkl", "rb") as f:
    loss_list = pickle.load(f)

#
plt.figure(figsize=(8, 6), dpi=300)
plt.plot([i+1 for i in range(len(loss_list))], loss_list, linewidth=2, label="Discriminator Loss", c="#009933")
plt.plot([i+1 for i in range(len(loss_list))], g_list, linewidth=2, label="Generator Loss", c="#0000FF")

plt.ylabel("Loss", fontsize=16, weight='semibold')
plt.yticks(fontsize=14, weight='semibold')
plt.xticks(fontsize=14, weight='semibold')
plt.xlabel("Time", fontsize=16, weight='semibold')
plt.legend(fontsize=14, prop={'size': 14, 'weight':'semibold'})
plt.savefig("loss.png")

plt.show()

plt.figure(figsize=(8, 6), dpi=300)
plt.plot([i+1 for i in range(len(loss_list))], loss_list, linewidth=2, label="Discriminator Loss", c="#009933")

plt.ylabel("Loss", fontsize=16, weight='semibold')
plt.yticks(fontsize=14, weight='semibold')
plt.xticks(fontsize=14, weight='semibold')
plt.xlabel("Time", fontsize=16, weight='semibold')
plt.legend(fontsize=14, prop={'size': 14, 'weight':'semibold'})
plt.savefig("loss1.png")

plt.show()
#
plt.figure(figsize=(8, 6), dpi=300)
plt.plot([i+1 for i in range(len(loss_list))], g_list, linewidth=2, label="Generator Loss", c="#0000FF")

plt.ylabel("Loss", fontsize=16, weight='semibold')
plt.yticks(fontsize=14, weight='semibold')
plt.xticks(fontsize=14, weight='semibold')
plt.xlabel("Time", fontsize=16, weight='semibold')
plt.legend(fontsize=14, prop={'size': 14, 'weight':'semibold'})
plt.savefig("loss2.png")

plt.show()