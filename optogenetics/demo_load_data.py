# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: APT
#     language: python
#     name: python3
# ---

# %%
# import 

# %load_ext autoreload
# %autoreload 2

import movies
import TrkFile
import numpy as np
import os
import matplotlib.pyplot as plt



# %%
# locations of data

#rootdatadir = '/groups/branson/bransonlab/flydisco_data'
rootdatadir = '/groups/branson/bransonlab/flybubble_social'
expname = 'socialCsChr_JRC_SS57589_CsChrimson_RigC_20190910T165024'
moviefilestr = 'movie.ufmf'
flytrackerfilestr = 'registered_trx.mat'
indicatorfilestr = 'indicatordata.mat'
aptfilestr = 'apt.trk'
perframedirstr = 'perframe'

# %%
# open movie, read a frame

f = 123
expdir = os.path.join(rootdatadir, expname)
moviefilename = os.path.join(expdir, moviefilestr)
assert os.path.exists(moviefilename), f'Movie file {moviefilename} does not exist'
movieobj = movies.Movie(moviefilename)
movieobj.open()
im,timestamp = movieobj.get_frame(f)

fig,ax = plt.subplots(1,1, figsize=(12,12))
ax.imshow(im, cmap='gray', vmin=0, vmax=255)
ax.set_title(f'Frame {f}, timestamp {timestamp:.3f} s')
movieobj.close()

# %%
# load body tracking data
flytrackerfile = os.path.join(expdir, flytrackerfilestr)
assert os.path.exists(flytrackerfile), f'Flytracker file {flytrackerfile} does not exist'
print(f'Loading flytracker data from {flytrackerfile}')
trx = TrkFile.load_trx(flytrackerfile)

# %%
# plot ellipses for all flies in the frame
# plot wings as two lines from centroid
fig,ax = plt.subplots(1,1, figsize=(12,12))
ax.imshow(im, cmap='gray', vmin=0, vmax=255)
colors = plt.cm.get_cmap('hsv', trx['ntargets'])

for fly in range(trx['ntargets']):
    if f < trx['firstframe'][fly] or f > trx['endframe'][fly]:
        continue
    i = f - trx['firstframe'][fly]
    x = trx['x'][fly][i]
    y = trx['y'][fly][i]
    a = trx['a'][fly][i]
    b = trx['b'][fly][i]
    theta = trx['theta'][fly][i]
    ell = plt.matplotlib.patches.Ellipse((x,y), 4*a, 4*b, angle=np.degrees(theta),
                                        edgecolor=colors(fly), facecolor='none', lw=2)
    ax.add_patch(ell)
    xwingl = trx['xwingl'][fly][i]
    ywingl = trx['ywingl'][fly][i]
    xwingr = trx['xwingr'][fly][i]
    ywingr = trx['ywingr'][fly][i]
    ax.plot([xwingl,x,xwingr], [ywingl,y,ywingr], color=colors(fly), lw=2)
ax.set_title(f'Frame {f}, timestamp {timestamp:.3f} s')

# plot speed of each fly in mm/s
maxspeed = 50 # mm/s
fig,ax = plt.subplots(trx['ntargets'],1, figsize=(8,1*trx['ntargets']),
                      sharex=True, sharey=True)
speed = []
for fly in range(trx['ntargets']):
    x = trx['x_mm'][fly]
    y = trx['y_mm'][fly]
    dt = trx['dt'][fly]
    f0 = trx['firstframe'][fly]
    f1 = trx['endframe'][fly]    
    speed = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2) / dt
    ax[fly].plot(trx['timestamps'][f0:f1], speed, color=colors(fly))
ax[-1].set_ylabel('Speed (mm/s)')
ax[-1].set_xlabel('Time (s)')
ax[-1].set_ylim(0, maxspeed)
fig.tight_layout()

# %%
# load in apt tracking data
aptfile = os.path.join(expdir, aptfilestr)
assert os.path.exists(aptfile), f'APT file {aptfile} does not exist'
print(f'Loading APT data from {aptfile}')
apttrk = TrkFile.Trk(aptfile)

# plot keypoints for the current frame
fig,ax = plt.subplots(1,1, figsize=(12,12))
ax.imshow(im, cmap='gray', vmin=0, vmax=255)
p = apttrk.getframe(f) # p is nkeypoints x 2 x len(f) x nflies
for fly in range(apttrk.ntargets):
    ax.plot(p[:,0,0,fly], p[:,1,0,fly], '.', color=colors(fly), lw=2)

# %%
# load indicator data
indicatorfile = os.path.join(expdir, indicatorfilestr)
assert os.path.exists(indicatorfile), f'Indicator file {indicatorfile} does not exist'
print(f'Loading indicator data from {indicatorfile}')
inddata,_ = TrkFile.loadmat(indicatorfile)

# plot indicator data
fig,ax = plt.subplots(1,1, figsize=(12,4))
ax.plot(trx['timestamps'],inddata['indicatorLED']['indicatordigital'],'k.')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Indicator LED digital value')

# %%
# load a per-frame feature
perframedir = os.path.join(expdir, perframedirstr)
featurename = 'dnose2ell'
perframefile = os.path.join(perframedir, f'{featurename}.mat')
assert os.path.exists(perframefile), f'Per-frame feature file {perframefile} does not exist'
perframedata = TrkFile.load_perframedata(perframefile)

max_timestamp = trx['timestamps'][-1]
T = np.max(trx['endframe'])+1

# plot per-frame feature with with indicator data
fig,ax = plt.subplots(1,1, figsize=(12,4))

# plot per-frame data
mean_pd = np.zeros(T)
count = np.zeros(T)
for fly in range(trx['ntargets']):
    datacurr = perframedata[fly] # data is 1 x T_fly
    f0 = trx['firstframe'][fly]
    n = datacurr.size
    mean_pd[f0:f0+n] += datacurr
    count[f0:f0+n+1] += 1
mean_pd /= np.maximum(count,1)
mean_pd[count==0] = np.nan

maxval = np.nanmax(mean_pd)*1.1

# plot indicator data as patches
for st,et in zip(inddata['indicatorLED']['starttimes'].flatten(), inddata['indicatorLED']['endtimes'].flatten()):
    # draw a rectangle from st to et, height 0 to maxval
    print(st,et)
    rect = plt.matplotlib.patches.Rectangle((st,0), et-st, maxval,
                                           facecolor='red', alpha=0.3)
    ax.add_patch(rect)   

ax.plot(trx['timestamps'], mean_pd, 'k-', lw=2)
ax.set_ylabel(featurename)
ax.set_xlabel('Time (s)')


# %%
# plot minimum distance between flies based on all keypoints
mean_mind = np.zeros(apttrk.T1+1)
count = np.zeros(apttrk.T1+1)
for fly in range(apttrk.ntargets):
    mind = np.inf
    pcurr = apttrk.gettarget(fly)[...,0] # pcurr is nkeypoints x 2 x T
    for otherfly in range(apttrk.ntargets):
        if otherfly == fly:
            continue
        pother = apttrk.gettarget(otherfly)[...,0]
        # compute all keypoint pairwise distances d which is nkeypoints x nkeypoints x T
        dcurr = np.min(np.sum((pcurr[None] - pother[:,None])**2,axis=2),axis=(0,1)) # d is nkeypoints x nkeypoints x T
        mind = np.minimum(mind, dcurr)
    isvalid = TrkFile.real_idx(pcurr)
    mean_mind[isvalid] += mind[isvalid]
    count[isvalid] += 1

mean_mind /= np.maximum(count,1)
mean_mind[count==0] = np.nan

fig,ax = plt.subplots(1,1, figsize=(12,4))
maxval = np.nanmax(mean_mind)*1.1

# plot indicator data as patches
for st,et in zip(inddata['indicatorLED']['starttimes'].flatten(), inddata['indicatorLED']['endtimes'].flatten()):
    # draw a rectangle from st to et, height 0 to maxval
    print(st,et)
    rect = plt.matplotlib.patches.Rectangle((st,0), et-st, maxval,
                                           facecolor='red', alpha=0.3)
    ax.add_patch(rect)   

ax.plot(trx['timestamps'], mean_mind, 'k-', lw=2)
