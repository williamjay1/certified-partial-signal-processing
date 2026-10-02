"""Original Hydraulic real statistic/spectral feature operator; no Torch import.

No preparation, fitting, prediction, or timing is run by importing this module.
"""
from __future__ import annotations
import numpy as np

SENSORS = ['PS1','PS2','PS3','PS4','PS5','PS6','EPS1','FS1','FS2','TS1','TS2','TS3','TS4','VS1']
STATS = ['mean','std','min','max','q25','q50','q75','first','last','slope_per_cycle','rms']
SPECTRAL = ['log_power','centroid_hz','spread_hz','spectral_entropy','dominant_hz','low_third_power_fraction']
WIDTHS = np.tile([11, 6], 14)
OFFSETS = np.r_[0, np.cumsum(WIDTHS)]
GROUP_INDEX = np.repeat(np.arange(28), WIDTHS)
SEED = 20261002

class RawBlocks:
    def __init__(self, raw, center=None, scale=None):
        self.raw = raw
        self.center = np.zeros(238, np.float32) if center is None else np.asarray(center, np.float32)
        self.scale = np.ones(238, np.float32) if scale is None else np.asarray(scale, np.float32)
        self.t = [np.linspace(-.5, .5, a.shape[1]) for a in raw]
        self.tt = [float(t@t) for t in self.t]
        self.freq = [np.fft.rfftfreq(a.shape[1], d=60./a.shape[1]) for a in raw]

    def values(self, case, block):
        j = block // 2
        x = self.raw[j][case]
        if block % 2 == 0:
            q = np.quantile(x, [.25,.5,.75])
            out = np.asarray([x.mean(),x.std(),x.min(),x.max(),*q,x[0],x[-1],
                              float(x@self.t[j])/self.tt[j],np.sqrt(np.mean(x*x))],np.float32)
        else:
            power = np.abs(np.fft.rfft(x-x.mean()))**2
            total = float(power.sum())
            norm = power / max(total, 1e-30)
            freq = self.freq[j]
            centroid = float(norm@freq)
            spread = np.sqrt(float(norm@((freq-centroid)**2)))
            ent = float(-(norm*np.log(np.maximum(norm,1e-30))).sum()/np.log(max(len(norm),2)))
            out = np.asarray([np.log1p(total/len(x)**2),centroid,spread,ent,
                              freq[int(np.argmax(power))],norm[freq<=freq[-1]/3].sum()],np.float32)
        return out

    def reveal(self, case, block, obs, mask):
        if mask[block] > .5: raise AssertionError('A block must not be computed twice')
        sl = slice(int(OFFSETS[block]), int(OFFSETS[block+1]))
        obs[sl] = (self.values(case,block)-self.center[sl])/self.scale[sl]
        mask[block] = 1
