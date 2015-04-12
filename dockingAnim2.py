# Benjamin Shih benjshih@gmail.com
# Incorporated one-dimensional damping using closed form differential equation solutions.

# Original mass-spring vibration.py code by Stefan van der Walt <stefan@sun.ac.za>, 2013
# License: CC0
# https://github.com/stefanv/vibrations/blob/master/vibrations.py

from __future__ import division

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle


# Workaround to matplotlib bug in FunctionAnimator that won't allow us
# to specify ``itertools.count(step=delta)`` as the frames argument.
# Also see ``set_time`` below.
delta = 0.02
PLAT_HEIGHT = 0.1
PLAT_WIDTH = 0.05
CUBE_HEIGHT = 0.08
CUBE_WIDTH = 0.08
XLIMMIN = 0
XLIMMAX = 2
YLIMMIN = -0.3
YLIMMAX = 0.2
PLAT_MASS = 1
PLAT_CUBE = 1
pause = False



class FreeUndamped(object):
    def __init__(self, k, m, x0, x0_prime):
        self.w0 = np.sqrt(k / m)
        self.B = x0
        self.A = x0_prime / self.w0

    def __call__(self, t):
        A, B, w0 = self.A, self.B, self.w0
        return A * np.sin(w0 * t) + B * np.cos(w0 * t)

class FreeUnderDamped(object):
    def __init__(self, k, m, c, x0, x0_prime):
        self.zeta = c/(2*np.sqrt(k*m)) # damping ratio.
        self.wn = np.sqrt(k/m) # natural frequency.
        self.wd = self.wn * np.sqrt(1-np.square(self.zeta)) # damped natural frequency.
        self.A = np.sqrt(np.square((x0*self.wd)) + np.square((x0_prime + x0*self.zeta*self.wn)))/self.wd # amplitude.
        self.phi = np.arctan2((x0*self.wd), (x0_prime+x0*self.zeta*self.wn)) # phase shift. 

    def __call__(self, t):
        zeta, wn, wd, A, phi = self.zeta, self.wn, self.wd, self.A, self.phi
        xpos = A*np.exp(-zeta*wn*t)*np.sin(wd*t+phi)
        return xpos

class FreeCriticallyDamped(object):
    def __init__(self, k, m, c, x0, x0_prime):
        self.zeta = c/(2*np.sqrt(k*m)) # damping ratio.
        self.c1 = x0
        self.c2 = x0_prime + x0*self.zeta*wn

    def __call__(self, t):
        c1, c2 = self.c1, self.c2
        xpos = c1*np.exp(-zeta*wn*t) + c2*t*np.exp(-zeta*wn*t)
        return xpos

class FreeOverDamped(object):
    def __init__(self, k, m, c, x0, x0_prime):
        self.zeta = c/(2*np.sqrt(k*m)) # damping ratio.
        self.wn = np.sqrt(k/m) # natural frequency.
        self.c1 = (x0*self.wn*(np.sqrt(np.square(self.zeta) - 1) + self.zeta) + x0_prime) / (2*self.wn*np.sqrt(np.square(self.zeta) - 1))
        self.c2 = (x0*self.wn*(np.sqrt(np.square(self.zeta) - 1) - self.zeta) - x0_prime) / (2*self.wn*np.sqrt(np.square(self.zeta) - 1))
        self.lambda1 = -self.zeta*self.wn + self.wn*np.sqrt(np.square(self.zeta) - 1)
        self.lambda2 = -self.zeta*self.wn - self.wn*np.sqrt(np.square(self.zeta) - 1)

    def __call__(self, t):
        c1, c2, lambda1, lambda2 = self.c1, self.c2, self.lambda1, self.lambda2
        xpos = c1*np.exp(lambda1*t) + c2*np.exp(lambda2*t)
        return xpos


class Spring(object):
    N = 100
    _hist_length = 100

    _spring_coords = np.zeros(N)
    _spring_coords[30:70] = 0.05 * (-1) ** np.arange(40)

    def __init__(self, axis, axis_history, k, m, c, F, x0, x0_prime):
        zeta = c/(2*np.sqrt(k*m))
        print zeta
        if c == 0 or F == 0:
            self._model = FreeUndamped(k, m, x0, x0_prime)
        elif zeta < 1:
            self._model = FreeUnderDamped(k, m, c, x0, x0_prime)
        elif zeta == 1:
            self._model = FreeCriticallyDamped(k, m, c, x0, x0_prime)
        elif zeta > 1:
            self._model = FreeOverDamped(k, m, c, x0, x0_prime)
        else:
            raise NotImplementedError()

        self._t = 0
        self._anchor = axis.vlines([0], -0.1, 0.1, linewidth=5, color='black')
        self._pot = Rectangle((self.u-PLAT_WIDTH, -PLAT_HEIGHT/2), PLAT_WIDTH, PLAT_HEIGHT, color='black')
        self._pot2 = Rectangle((self.u+CUBE_WIDTH, -CUBE_HEIGHT/2), CUBE_WIDTH, CUBE_HEIGHT, color='red')
        # self._pot = Circle((self.u, 0), 0.05, color='black')
        self._spring, = axis.plot(*self._spring_xy(), color='black')

        axis.vlines([1], -0.1, -0.2)
        axis.text(1, -0.25, '$x = 0$', horizontalalignment='center')

        self._ax = axis
        axis.add_patch(self._pot)
        axis.add_patch(self._pot2)
        axis.set_xlim([XLIMMIN, XLIMMAX])
        axis.set_ylim([YLIMMIN, YLIMMAX])
        axis.set_axis_off()
        axis.set_aspect('equal')

        self._history = [self.u - 1] * self._hist_length
        self._history_plot, = axis_history.plot(np.arange(self._hist_length) * delta, self._history)
        axis_history.annotate('Now',
                              (self._hist_length * delta, 1.5),
                              (self._hist_length * delta, 1.8),
                              arrowprops=dict(arrowstyle='->'),
                              horizontalalignment='center')
        axis_history.set_ylim(-2, 1.5)
        axis_history.set_xticks([])
        axis_history.set_xlabel(r'$\mathrm{Time}$')
        axis_history.set_ylabel(r'$\mathrm{Position,\, x}$')

    def _spring_xy(self):
        return np.linspace(0, self.u, self.N), self._spring_coords

    def set_time(self, t):
        if not pause:
            self._t = t * delta
            self.update()
        # yield self._t

    @property
    def u(self):
        # print self._t
        # print self._model(self._t)
        return 1 + self._model(self._t)

    def update(self):
        # self._pot.center = (self.u, 0)
        self._pot.set_x(self.u-PLAT_WIDTH)
        self._pot2.set_x(self.u)
        
        x, y = self._spring_xy()
        self._spring.set_xdata(x)

        self._history.append(self.u - 1)
        self._history = self._history[-self._hist_length:]

        self._history_plot.set_ydata(self._history)

class IncomingCube(object):
    def __init__(self, axis, m, x0, x0_prime):
        self._t = 0
        self._pot = Rectangle((x0, -CUBE_HEIGHT/2), CUBE_WIDTH, CUBE_HEIGHT, color='red')

        self._ax = axis
        axis.add_patch(self._pot)
        self._cubePos = CubePhysics(m, x0, x0_prime)

    def set_time(self, t):
        if not pause:
            self._t = t * delta
            self.update()
        # yield self._t

    @property
    def u(self):
        return 1 + self._cubePos(self._t)

    def update(self):
        self._pot.set_x(self.u-CUBE_WIDTH)


class CubePhysics(object):
    def __init__(self, m, x0, x0_prime):
        pass

    def __call__(self, t):
        return x0 + x0_prime*t + 0.5

def onClick(event):
    global pause
    pause ^= True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Illustration simple mechanical vibrations')
    parser.add_argument('-o', '--output',
                        help='Filename for output video.  Requires ffmpeg.')
    parser.add_argument('-m', '--mass', type=float, default=PLAT_CUBE)
    parser.add_argument('-c', '--damping', type=float, default=0.75)
    parser.add_argument('-k', '--spring', type=float, default=1)
    parser.add_argument('-F', '--force', type=float, default=1)
    parser.add_argument('--x0', type=float, default=0)
    parser.add_argument('--x0_prime', type=float, default=-0.2)
    args = parser.parse_args()

    m, c, k, F, x0, x0_prime = (getattr(args, name) for name in
                                    ('mass', 'damping', 'spring', 'force',
                                     'x0', 'x0_prime'))

    f, (ax0, ax1) = plt.subplots(2, 1)
    s = Spring(axis=ax0, axis_history=ax1, k=k, m=PLAT_MASS, c=c, F=F, x0=x0, x0_prime=x0_prime)
    # cube = IncomingCube(axis=ax0, m=m, x0=XLIMMAX*0.5, x0_prime=-x0_prime)

    anim = animation.FuncAnimation(f, s.set_time, interval=delta * 1000, save_count=400)

    if args.output:
        print "Saving video output to %s (this may take a while)." % args.output
        anim.save(args.output, fps=25)

    plt.show()
