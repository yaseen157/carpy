from functools import partial

from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.widgets import RadioButtons, RangeSlider
import numpy as np

from carpy.utility import Quantity

__all__ = ["TimeSeries"]


class TimeSeries:

    def __init__(self, _name, /, timestamps, **data):
        self._name = _name
        self._time = Quantity(timestamps, "s")
        self._data = data
        return

    def inspect(self):
        """Start an interactive data visualisation window."""

        def s2d(seconds: float) -> float:
            """Convert from seconds (float) to days."""
            return seconds / 86_400

        # Create figure and axes
        n_plots = len(self._data)
        fig, axs = plt.subplot_mosaic(
            [
                [f"data:{i}", f"radio:{i}"]
                for i in range(n_plots)
            ],
            figsize=(10, 6),
            width_ratios=[5, 1],
            num=f"{type(self).__name__}:{self._name}",
        )

        fig.subplots_adjust(left=0.15, bottom=0.23, right=0.94, wspace=0.015)
        radios, lines = dict(), dict()
        qty2lbl = {
            'kg m^{-1} s^{-2}': [
                "Pa", "hPa", "mbar", "$lbf$ $ft^{-2}$", "inHg"],
            'K': ["K", "degC", "degF", "degR"]
        }

        # Initially plot all the data
        for i, (key, val) in enumerate(self._data.items()):
            ax_data = axs[f"data:{i}"]
            ax_radio = axs[f"radio:{i}"]

            # Matplotlib plots dates as the float number of days (convert secs)
            lines[i], = ax_data.plot(s2d(self._time), val)

            # Identify the units if val is an instance of quantity
            if isinstance(val, Quantity) and val.units.si_equivalent in qty2lbl:

                # Save new radio menu
                radios[i] = RadioButtons(
                    ax=ax_radio,
                    labels=qty2lbl[val.units.si_equivalent]
                )

                # When selection is made on radio menu, rescale the y-axis
                def mapper(label, id, k):
                    """label: Radio label, id: row, k: self._data['key']."""
                    y_data = self._data[k].to(label)
                    lines[id].set_ydata(y_data)
                    delta_lim = ((t := y_data.max()) - (b := y_data.min())) / 20
                    axs[f"data:{id}"].set_ylim(b - delta_lim, t + delta_lim)
                    fig.canvas.draw_idle()
                    return

                radios[i].on_clicked(partial(mapper, id=i, k=key))

            else:
                ax_radio.get_xaxis().set_visible(False)
                ax_radio.get_yaxis().set_visible(False)

        # Labelling
        ax = plt.gca()  # Default to stop IDE sadness
        for i, key in enumerate(self._data.keys()):
            ax = axs[f"data:{i}"]

            # x-axis invisible, axes limits set by data bounds
            ax.get_xaxis().set_visible(False)
            ax.set_xlim(
                np.min(lines[i].get_xdata()),
                np.max(lines[i].get_xdata())
            )

            # y-axis labelling
            ax.ticklabel_format(useOffset=False)
            ax.yaxis.set_label_coords(-0.12, 0.5)
            ax.set_ylabel(key)

        else:
            # Lower left data axes needs x-labels so user can see timestamps
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel("Timestamp [HH:MM:SS]")
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))

            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment('right')

        # Spawn a new axes object, containing the slider element
        slider = RangeSlider(
            ax=plt.axes((0.2, 0.05, 0.60, 0.03)), label="Context [s]",
            valinit=(self._time.min(), self._time.max()),
            valmin=self._time.min(), valmax=self._time.max(),
            valfmt='%i'
        )

        def update(val) -> None:
            """Whenever the sliders are moved, call this. val: (min, max)."""
            # Update the selection
            for i in range(n_plots):
                ax = axs[f"data:{i}"]
                ax.axis([s2d(val[0]), s2d(val[1]), *(ax.get_ylim())])

            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
        return
