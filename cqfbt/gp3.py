import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title('Main Title')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Sub Title', loc='left', fontsize=10)

plt.show()