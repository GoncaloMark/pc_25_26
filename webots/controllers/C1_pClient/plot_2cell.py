import numpy as np
import matplotlib.pyplot as plt

# Tarkim turi duomenis


mean_right = 146.2547593027734
std_right = 4.700464785129498

mean_left = 67.63946300532409
std_left = 6.745633878497483

mean_front = 67.1338494743455 
std_front = 2.355019395182425

mean_back = 118.56428996878635
std_back = 5.3567224676935306

# Susidaryti x ašį aplink mean ±4*std
x_front = np.linspace(mean_front - 4*std_front, mean_front + 4*std_front, 1000)
x_back = np.linspace(mean_back - 4*std_back, mean_back + 4*std_back, 1000)
x_left = np.linspace(mean_left - 4*std_left, mean_left + 4*std_left, 1000)
x_right = np.linspace(mean_right - 4*std_right, mean_right + 4*std_right, 1000)

# Gaussian PDF formulė be scipy
def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Apskaičiuojam PDF kiekvienai pusei
pdf_front = gaussian_pdf(x_front, mean_front, std_front)
pdf_back = gaussian_pdf(x_back, mean_back, std_back)
pdf_left = gaussian_pdf(x_left, mean_left, std_left)
pdf_right = gaussian_pdf(x_right, mean_right, std_right)

# Grafikas
plt.plot(x_front, pdf_front, label='Front')
plt.plot(x_back, pdf_back, label='Back')
plt.plot(x_left, pdf_left, label='Left')
plt.plot(x_right, pdf_right, label='Right')
plt.xlabel('Sensor Reading with 2 cell apart')
plt.ylabel('Probability Density')
plt.title('Gaussian PDF of Sensor Measurements')
plt.legend()
plt.grid(True)
plt.show()