import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st

st.title("Lotka-Volterra-Modell – interaktive Simulation")

# Parameter-Slider in der Sidebar
a = st.sidebar.slider("Beute-Wachstumsrate (a)", 0.01, 1.0, 0.1, step=0.01)
alp = st.sidebar.slider("Räuber-Einfluss (α) in 10^(-4)", 0.1, 0.11, 2, step=0.01) * 10^(-4)
c = st.sidebar.slider("Räuber-Sterberate (c)", 0.01, 1.0, 0.4, step=0.01)
gam = st.sidebar.slider("Wachstum durch Beute (γ)", 0.00001, 0.001, 0.0002, step=0.00001)
x0 = st.sidebar.slider("Startpopulation Beute", 100, 2000, 1000, step=100)
y0 = st.sidebar.slider("Startpopulation Räuber", 100, 2000, 1000, step=100)

# Zeitintervall
t_span = (0, 140)
t_eval = np.linspace(*t_span, 10000)

# Differentialgleichungssystem
def f(t, z):
    x, y = z
    dx = x * (a - alp * y)
    dy = y * (-c + gam * x)
    return [dx, dy]

sol = solve_ivp(f, t_span, [x0, y0], t_eval=t_eval)

# Plot mit Matplotlib
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sol.t, sol.y[0], label='Beute', color='green')
ax.plot(sol.t, sol.y[1], label='Räuber', color='red')
ax.set_xlabel("Zeit")
ax.set_ylabel("Population")
ax.set_title("Lotka-Volterra-Modell")
ax.grid(True)
ax.legend()

# Anzeige in Streamlit
st.pyplot(fig)
