
#%%
import numpy as np
import matplotlib.pyplot as plt

# Sample data
categories = ['Irritation', 'Impatience', 'Surprise']
values1 = [22.86, 42.33, 30.09]
values2 = [45.07, 79, 86]
values3 = [23.33, 36.66, 43.33]

# Create bar graph
x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars
fig, ax = plt.subplots()
rects3 = ax.bar(x - width, values3, width, label='Surveys', color='white', edgecolor='black', hatch='--')
rects1 = ax.bar(x, values1, width, label='USC', color='white', edgecolor='black', hatch='\\\\')
rects2 = ax.bar(x + width, values2, width, label='TRINA', color='white', edgecolor='black', hatch='xx')

# # Highlight the best results
# best_values = np.maximum.reduce([values1, values2, values3])
# for i, rect in enumerate(rects1):
#     if values1[i] == best_values[i]:
#         rect.set_edgecolor('black')
#         rect.set_linewidth(2)
# for i, rect in enumerate(rects2):
#     if values2[i] == best_values[i]:
#         rect.set_edgecolor('black')
#         rect.set_linewidth(2)
# for i, rect in enumerate(rects3):
#     if values3[i] == best_values[i]:
#         rect.set_edgecolor('black')
#         rect.set_linewidth(2)

# Add title and labels
ax.set_title('Sample Bar Graph')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.show()
# %%
