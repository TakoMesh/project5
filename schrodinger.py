import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft,ifft


class Wavefunction(object):

    def __init__(self, x, psi, V, k0=None, h=1, m=1, Ti=0.0):
        self.x = x
        self.dx = self.x[1] - self.x[0]
        self.N = len(x)
        self.psi = psi
        self.V = V
        self.h = h
        self.m = m
        self.Ti = Ti
        self.dT = .0001
        self.dk = 2 * np.pi / (self.N * self.dx)
        if k0 is None:
            self.k0 = -np.pi / self.dx
        else:
            self.k0 = k0
        self.k = np.linspace(self.k0, self.k0+self.N*self.dk, self.N)

        self.psi_x = self.psi
        self.psi_k = fft(self.psi_x)

    def psi_k_plot(self):
        return np.concatenate((self.psi_k[self.N / 2:], self.psi_k[0:self.N / 2]))

    def halfstep_x(self, dT=None):
        if dT is None:
            dT = self.dT
        self.psi_x = np.multiply(self.psi_x, np.exp(-1j*(dT*self.V)/(2*self.h)))
        self.psi_k = fft(self.psi_x)

    def fullstep_k(self, dT=None):
        if dT is None:
            dT = self.dT
        self.psi_k = np.multiply(self.psi_k, np.exp(-1j*(dT*self.h*self.k**2)/(2*self.m)))
        self.psi_x = ifft(self.psi_k)

    def timestep(self, dT=None):
        for i in xrange(10):
            self.halfstep_x(dT)
            self.fullstep_k(dT)
            self.halfstep_x(dT)


# - - - TEST:

def test():
    ts = 1000   # Time steps
    kr = 100000    # K-space range
    x = np.linspace(-100, 100, 10000)
    v = np.zeros(x.shape)
    v[x > 5] = 5000.0
    v[x > 5.2] = 0.0
    v[x < -10] = 100000.0
    v[x > 10] = 100000.0
    wf = Wavefunction(x, 0.5*np.exp(-x**2 - 1j*25*x), v)

    fig = plt.figure()
    ax1 = fig.add_subplot(211, xlim=(-11, 11), ylim=(0, 1))
    coordinate, = ax1.plot(wf.x, np.abs(wf.psi_x))
    potential, = ax1.plot(wf.x, wf.V)
    ax2 = fig.add_subplot(212, xlim=(-150, 150))
    momenta, = ax2.plot(wf.k, np.abs(wf.psi_k_plot()), color='r')
    plt.grid()

    def init():
        coordinate.set_data(wf.x, np.abs(wf.psi_x))
        momenta.set_data(wf.k, np.abs(wf.psi_k_plot()))
        potential.set_data(wf.x, wf.V)
        return coordinate, momenta, potential,

    def animate(i):
        wf.timestep(0.0001)
        coordinate.set_data(wf.x, np.abs(wf.psi_x))
        momenta.set_data(wf.k, np.abs(wf.psi_k_plot()))
        potential.set_data(wf.x, wf.V)
        return coordinate, momenta, potential,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=ts, interval=1, blit=True)
    # anim.save('schrodinger_barrier.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

    plt.show()

test()