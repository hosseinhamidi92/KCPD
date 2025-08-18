import numpy as np

#%%

#
import matplotlib.pyplot as plt

# Dummy data
categories = ['Slow Car', 'Brake 1', 'Brake 2', 'Brake 3', 'Brake 4','Brake 5']
subcategories = ['Engaged', 'Evaded']
data = np.array([[11, 4],
       [4, 11],
       [5, 10],
       [6,9],
       [13, 2],
       [2, 13]])

# Plotting the stacked bar graph
fig, ax = plt.subplots()

for i in range(len(subcategories)):
    colors = ['seagreen', 'red']
    ax.bar(categories, data[:, i], bottom=np.sum(data[:, :i], axis=1), label=subcategories[i], color=colors[i], alpha=0.7)

ax.set_xlabel('Events')
ax.set_ylabel('Number of Occurrence')
# ax.set_title('Stacked Bar Graph with Dummy Data')
ax.legend(title='Interaction', bbox_to_anchor=(1.05, 1), loc='upper left')

ax.set_yticks(np.arange(0, np.max(np.sum(data, axis=1)) + 1, 1))
ax.yaxis.set_major_locator(plt.MultipleLocator(3))
fig.tight_layout()
plt.show()

# %%
