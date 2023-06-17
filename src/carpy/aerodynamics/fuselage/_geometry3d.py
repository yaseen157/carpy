"""Methods for modelling the 3d geometry of a fuselage."""
import numpy as np
import streamlit as st


class SuperEllipse(object):

    def __init__(self):
        # Position offset
        self._posX = 0
        self._posY = 0

        # Shape parameters (default for a circle)
        self._m = 2  # Default for an ellipse-type shape
        self._n = 1  # Default for an ellipse-type shape
        self._a = 1
        self._b = 1
        return

    def plot(self, N: int):
        # T is a parameter with no physical interpretation
        t = np.linspace(0, np.pi / 2, N)

        x = self._a * np.cos(t) ** (2 / self._m)
        y = self._b * np.sin(t) ** (2 / self._n)

        from matplotlib import pyplot as plt

        plt.plot(x, y)
        plt.plot(x, -y)
        plt.plot(-x, -y)
        plt.plot(-x, y)
        plt.show()


if __name__ == "__main__":
    # import streamlit as st

    age = st.slider('How old are you?', 0, 130, 25)
    st.write("I'm ", age, 'years old')

    # Streamlit runs from the terminal. Have to do: streamlit "path-to-file.py"

