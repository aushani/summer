import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': '18',
         'axes.titlesize':'18',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small'}

x = np.arange(20, 2000)
y = (1 + 0.7 * np.sin(10*np.log(x))) * (x**2)

ticks = np.asarray([20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000])
ticks_baba = np.asarray([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
ticks_arash = np.asarray([20, 98, 152, 187, 987, 1342, 2000])

f, axarr = plt.subplots(nrows = 3, ncols = 1)

axarr[0].loglog(x, y)
axarr[0].xaxis.set_ticks(ticks_baba)
axarr[0].xaxis.set_minor_locator(ticker.FixedLocator([0]))
axarr[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
axarr[0].grid(True)
axarr[0].set_title('Plot 1 (Analog Arts)')
axarr[0].axis([20, 2000, 1e2, 1e7])

axarr[1].loglog(x, y)
axarr[1].xaxis.set_ticks(ticks)
axarr[1].xaxis.set_minor_locator(ticker.FixedLocator([0]))
axarr[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
axarr[1].grid(True)
axarr[1].set_title('Plot 2')
axarr[1].axis([20, 2000, 1e2, 1e7])

axarr[2].loglog(x, y)
axarr[2].xaxis.set_ticks(ticks_arash)
axarr[2].xaxis.set_minor_locator(ticker.FixedLocator(ticks_arash))
axarr[2].xaxis.set_minor_locator(ticker.FixedLocator([0]))
axarr[2].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
axarr[2].grid(True)
axarr[2].set_title('Plot 3')
axarr[2].axis([20, 2000, 1e2, 1e7])

plt.show()
