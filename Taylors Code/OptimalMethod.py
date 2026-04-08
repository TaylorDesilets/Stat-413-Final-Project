

# Finding the best optimization method based on optimize function- python equivalent to R optim() func

from scipy.optimize import minimize
from ReadFromCSV import load_data

result = minimize(load_data(), x0=[0, 0])
