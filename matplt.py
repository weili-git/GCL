import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y1 = [2, 4, 7, 3, 1]
y2 = [5, 7, 2, 4, 6]
plt.style.use("ggplot")

plt.plot(x, y1, label="line1")
plt.plot(x, y2, label="line2")

plt.title("My graph")
plt.xlabel("x")
plt.ylabel("y")

plt.legend(loc="best", title="Metric")

# plt.grid()
plt.show()



