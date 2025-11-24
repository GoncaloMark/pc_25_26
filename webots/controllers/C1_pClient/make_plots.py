import re
import numpy as np
import matplotlib.pyplot as plt

def parse_sensor_file(filename):
    """
    Parses sensor statistics from the text file and returns a dictionary with all wall types.
    """
    data = {}
    with open(filename, "r") as f:
        content = f.read()

    sections = content.split('---')
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        wall_type = lines[0].strip()
        sensor_stats = {}
        for line in lines[1:]:
            match = re.match(r'MEAN_(\w+): ([\d\.]+) \| STD_\1: ([\d\.]+)', line.strip())
            if match:
                direction = match.group(1).lower()
                mean = float(match.group(2))
                std = float(match.group(3))
                sensor_stats[direction] = {"mean": mean, "std": std}
        if sensor_stats:
            data[wall_type] = sensor_stats
    return data

sensor_data_all = parse_sensor_file("mean_std_sensors_1000.txt")

def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

for wall_type, stats in sensor_data_all.items():
    x_front = np.linspace(stats['front']['mean'] - 4*stats['front']['std'], stats['front']['mean'] + 4*stats['front']['std'], 1000)
    x_back = np.linspace(stats['back']['mean'] - 4*stats['back']['std'], stats['back']['mean'] + 4*stats['back']['std'], 1000)
    x_left = np.linspace(stats['left']['mean'] - 4*stats['left']['std'], stats['left']['mean'] + 4*stats['left']['std'], 1000)
    x_right = np.linspace(stats['right']['mean'] - 4*stats['right']['std'], stats['right']['mean'] + 4*stats['right']['std'], 1000)

    pdf_front = gaussian_pdf(x_front, stats['front']['mean'], stats['front']['std'])
    pdf_back = gaussian_pdf(x_back, stats['back']['mean'], stats['back']['std'])
    pdf_left = gaussian_pdf(x_left, stats['left']['mean'], stats['left']['std'])
    pdf_right = gaussian_pdf(x_right, stats['right']['mean'], stats['right']['std'])

    plt.figure(figsize=(8,5))
    plt.plot(x_front, pdf_front, label='Front')
    plt.plot(x_back, pdf_back, label='Back')
    plt.plot(x_left, pdf_left, label='Left')
    plt.plot(x_right, pdf_right, label='Right')
    plt.xlabel('Sensor Reading')
    plt.ylabel('Probability Density')
    plt.title(f'Gaussian PDF of Sensor Measurements\n({wall_type.lower()})')
    plt.legend()
    plt.grid(True)
    plt.show()
