import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matio
import TrkFile
import os
import movies

expdir = '/groups/branson/bransonlab/datasets/flyopto/VNC2_Control_RigA_20230913T100524'
figdir = '/groups/branson/home/bransonk/behavioranalysis/code/ScienceBenchmark/optogenetics/figs'
basefigsize = 5

# %% show fly body tracking

moviefile = os.path.join(expdir, 'movie.ufmf')
movie = movies.Movie(moviefile)

# load tracking
print('Loading registered_trx.mat...')
trxfile = os.path.join(expdir,'registered_trx.mat')
trxdata = TrkFile.load_trx(trxfile)
print('Done.')

# colors for each fly
colors = plt.cm.tab10.colors
nflies = len(trxdata['x'])

fig,ax = plt.subplots(1,3, figsize=(1.5*basefigsize,.5*basefigsize))

f = 6760
im,ts = movie.get_frame(f)
tplot = 75

zoompt = (850,470)
zoomr = 150

colori = 0
him = []
for i in range(3):
    ax[i].cla()
    him.append(ax[i].imshow(im, vmin=0, vmax=255, cmap='gray'))

xzoom = np.array([zoompt[0]-zoomr, zoompt[0]+zoomr])
yzoom = np.array([zoompt[1]-zoomr, zoompt[1]+zoomr])
ax[0].plot(xzoom[[0,0,1,1,0]], yzoom[[0,1,1,0,0]], 'r-')
    
for i in range(nflies):
    if f < trxdata['firstframe'][i] or f > trxdata['endframe'][i]:
        continue
    j = f - trxdata['firstframe'][i]
    j0 = max(0, j - tplot)
    x = trxdata['x'][i][j]
    y = trxdata['y'][i][j]
    theta = trxdata['theta'][i][j]
    a = trxdata['a'][i][j]
    b = trxdata['b'][i][j]
    color = colors[colori]
    #ax[0].text(x,y,str(i),color=color,fontsize=12)
    # plot trajectory
    if j > j0:

        # plot trajectory with fading alpha
        traj_x = trxdata['x'][i][j0:j+1]
        traj_y = trxdata['y'][i][j0:j+1]
        n_pts = len(traj_x)
        points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0, .75, n_pts)  
        rgba_colors = [color[:3] + (a,) for a in alphas[:-1]]
        lc = LineCollection(segments, colors=rgba_colors, linewidths=1)
        ax[2].add_collection(lc)
        ax[2].scatter(traj_x, traj_y, c=[color[:3] + (a,) for a in alphas], s=1)

    #ax[2].plot(trxdata['x'][i][j0:j+1], trxdata['y'][i][j0:j+1], '.-', color=color[:3]+(0.5,), lw=1)
    # plot body ellipse
    ellipse = matplotlib.patches.Ellipse((x,y), 4*a, 4*b, angle=np.degrees(theta), edgecolor=color, facecolor='none', lw=2)
    ax[2].add_patch(ellipse)
    # plot heading line
    xh = x + 2 * a * np.cos(theta)
    yh = y + 2 * a * np.sin(theta)
    ax[2].plot(xh,yh, 'o',color=color)
    # plot center point
    ax[2].plot(x, y, 'o', color=color, markersize=2)
    colori += 1
        
for i in range(3):
    ax[i].set_aspect('equal')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].invert_yaxis()

ax[0].set_xlim(0, im.shape[1])
ax[0].set_ylim(0, im.shape[0])
for k in [1,2]:
    ax[k].set_xlim(zoompt[0]-zoomr, zoompt[0]+zoomr)
    ax[k].set_ylim(zoompt[1]-zoomr, zoompt[1]+zoomr)

fig.tight_layout()

# save to svg, pdf, png
expstr = os.path.basename(expdir)
outfilestr = os.path.join(figdir, f'fly_tracking_{expstr}_{f:05d}')
fig.savefig(outfilestr+'.svg')
fig.savefig(outfilestr+'.png', dpi=300)
fig.savefig(outfilestr+'.pdf')

# %% show registration

regdata,_ = matio.loadmat(os.path.join(expdir,'registrationdata.mat'))
fig,ax = plt.subplots(1,1, figsize=(.5*basefigsize,.5*basefigsize))
ax.cla()
ax.imshow(im, vmin=0, vmax=255, cmap='gray')
circlecenter = (regdata['circleCenterX'], regdata['circleCenterY'])
circleradius = regdata['circleRadius']
circle = matplotlib.patches.Circle(circlecenter, circleradius, edgecolor='r', facecolor='none', lw=2)
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()

# plot scale bar
pxpermm = 1./regdata['scale'].item()
scalebarlength_mm = 5  # 5 mm
scalebarlength_px = scalebarlength_mm * pxpermm
scalebar_x = im.shape[1] - 50 - scalebarlength_px
scalebar_y = im.shape[0] - 30
ax.plot([scalebar_x, scalebar_x + scalebarlength_px], [scalebar_y, scalebar_y], 'r-', lw=2)
ax.text(scalebar_x + scalebarlength_px/2, scalebar_y - 10, f'{scalebarlength_mm} mm', color='r', ha='center', va='bottom', fontsize=12)

# save to svg, pdf, png
outfilestr = os.path.join(figdir, f'registration_{expstr}_{f:05d}')
fig.savefig(outfilestr+'.svg')
fig.savefig(outfilestr+'.png', dpi=300)
fig.savefig(outfilestr+'.pdf')

# %% show keypoint tracking

keyptfile = os.path.join(expdir,'apt.trk')
trkdata = TrkFile.Trk(keyptfile)
fig,ax = plt.subplots(1,1, figsize=(.75*basefigsize,.75*basefigsize))
tplot = 10
ax.cla()
ax.imshow(im, vmin=0, vmax=255, cmap='gray')
nflies = trkdata.ntargets
colori = 0
for i in range(nflies):
    if f < trkdata.startframes[i] or f > trkdata.endframes[i]:
        continue
    color = colors[colori]
    
    # plot trajectory with fading alpha
    ptraj = trkdata.gettargetframe(i,np.arange(f-tplot,f+1))
    ptraj = ptraj[...,0]
    n_pts = ptraj.shape[2]
    alphas = np.linspace(0, .5, n_pts)  
    rgba_colors = [color[:3] + (a,) for a in alphas[:-1]]

    for k in range(ptraj.shape[0]):
        points = ptraj[k].T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=rgba_colors, linewidths=1)
        ax.add_collection(lc)
        ax.scatter(ptraj[k][0], ptraj[k][1], c=[color[:3] + (a,) for a in alphas], s=1)
    
    p = trkdata.gettargetframe(i,f)
    p = p[:,:,0,0]
    ax.plot(p[:,0], p[:,1], 'o', markersize=4)
    colori += 1

ax.set_xlim(zoompt[0]-zoomr, zoompt[0]+zoomr)
ax.set_ylim(zoompt[1]-zoomr, zoompt[1]+zoomr)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
#ax.invert_yaxis()

# save to svg, pdf, png
outfilestr = os.path.join(figdir, f'kpttracking_{expstr}_{f:05d}')
fig.savefig(outfilestr+'.svg')
fig.savefig(outfilestr+'.png', dpi=300)
fig.savefig(outfilestr+'.pdf')

# %% indicator detection

f = 8383
im,ts = movie.get_frame(f)

indfile = os.path.join(expdir,'indicatordata.mat')
inddata,_ = matio.loadmat(indfile)

fig, ax = plt.subplots(2, 1, figsize=(basefigsize/1.4,basefigsize), gridspec_kw={'height_ratios': [3, 1]})
ax[0].cla()
ax[0].imshow(im, vmin=0, vmax=255, cmap='gray')

x = inddata['rect'][0,0]
y = inddata['rect'][1,0]
w = inddata['rect'][2,0]
h = inddata['rect'][3,0]
ax[0].plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], 'r-')
ax[0].invert_yaxis()
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].cla()
ax[1].plot(inddata['indicatorLED']['indicatordigital'].flatten(),'k-')
ax[1].set_xlabel('Frame')
ax[1].set_yticks([0,1])
ax[1].set_yticklabels(['Off','On'])
ax[1].set_ylabel('Stimulus state')
fig.tight_layout()

# save to svg, pdf, png
outfilestr = os.path.join(figdir, f'stimulusstate_{expstr}_{f:05d}')
fig.savefig(outfilestr+'.svg')
fig.savefig(outfilestr+'.png', dpi=300)
fig.savefig(outfilestr+'.pdf')

# %% per-frame feature

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(2*basefigsize, 2/3*basefigsize))
gs = gridspec.GridSpec(6, 3, width_ratios=[1,1,1],hspace=0)

ax = {}
ax['left'] = fig.add_subplot(gs[:, 0])  # span all rows on left
ax['left'].set_aspect('equal')
ax['left'].set_xticks([])
ax['left'].set_yticks([])

ax['middle'] = [fig.add_subplot(gs[0, 1])]
# Rest share x and y with the first
ax['middle'] += [fig.add_subplot(gs[i, 1], sharex=ax['middle'][0], sharey=ax['middle'][0]) for i in range(1, 6)]
ax['right'] = [fig.add_subplot(gs[0, 2])]
# Rest share x and y with the first
ax['right'] += [fig.add_subplot(gs[i, 2], sharex=ax['right'][0], sharey=ax['right'][0]) for i in range(1, 6)]

tplot = 75

legtips = [11,12,13,14,15,16]

swingstancedata,_ = matio.loadmat('swingstance.mat')
f = 6760
fly = 3
im = movie.get_frame(f)[0]

ax['left'].cla()
ax['left'].imshow(im, vmin=0, vmax=255, cmap='gray')
p = trkdata.gettargetframe(fly,f)
p = p[:,:,0,0]
ptraj = trkdata.gettargetframe(fly,np.arange(f-tplot,f+tplot+1))
ptraj = ptraj[...,0]
minv = np.min(p,axis=0)
maxv = np.max(p,axis=0)
center = (minv + maxv) / 2
r = 75
ax['left'].set_xlim(center[0] - r, center[0] + r)
ax['left'].set_ylim(center[1] - r, center[1] + r)
FPS = 150

maxspeed = 0
for i,k in enumerate(legtips):
    ax['left'].plot(ptraj[k,0], ptraj[k,1], '.-', color=(colors[i][:3] + (0.75,)), markersize=1, lw=1)
    ax['left'].plot(p[k,0], p[k,1], 'o', markersize=5, color=colors[i])
    speed = np.linalg.norm(ptraj[k,:,1:] - ptraj[k,:,0:-1], axis=0)*FPS/pxpermm
    maxspeed = max(maxspeed, np.max(speed))
    ax['middle'][i].plot(np.arange(-tplot,tplot),speed, '.-', color=colors[i])

for i in range(6):
    ax['middle'][i].plot([0,0],[0,maxspeed], '--', lw=1, color=[0.5,0.5,0.5])
    ax['middle'][i].set_yticks(np.arange(0,maxspeed*.9,100))
ax['middle'][0].set_xlim(-tplot, tplot)
ax['middle'][0].set_ylim(0, maxspeed*1.1)

for i,k in enumerate(legtips):
    isswing = np.ones((1,2*tplot,3))
    color = colors[i]
    for c in range(3):
        isswing[0,swingstancedata['isswing'][f-tplot:f+tplot,i,fly]==1,c] = color[c]*.5+.5
        isswing[0,swingstancedata['isstance'][f-tplot:f+tplot,i,fly]==1,c] = color[c]*.5
    ax['right'][i].imshow(isswing, aspect='auto', extent=[-tplot, tplot, 0, 1])
    ax['right'][i].plot([0,0],[0,1], '--', lw=1, color=[0.5,0.5,0.5])
    ax['right'][i].set_yticks([])

for a in ax['middle'][:-1]:  # all except last
    a.tick_params(labelbottom=False)
for a in ax['right'][:-1]:  # all except last
    a.tick_params(labelbottom=False)

ax['left'].set_xticks([])
ax['left'].set_yticks([])

fig.tight_layout()

# save to svg, pdf, png
outfilestr = os.path.join(figdir, f'swingstance_{expstr}_{f:05d}')
fig.savefig(outfilestr+'.svg')
fig.savefig(outfilestr+'.png', dpi=300)
fig.savefig(outfilestr+'.pdf')

# %% behavior classification
scoresfile = os.path.join(expdir,'scores_Walk2.mat')
sd,_ = matio.loadmat(scoresfile)
iswalking = np.ones((nflies,tplot*2+1,3))
colori = 0
for fly in range(nflies):
    color = colors[colori]
    for c in range(3):
        tf = sd['allScores']['postprocessed'][fly][f-tplot:f+tplot+1,0] == 1
        iswalking[fly,tf,c] = color[c]
    colori += 1
fig,ax = plt.subplots(1,1, figsize=(.75*basefigsize,.5*basefigsize))
ax.cla()
ax.imshow(iswalking, aspect='auto', extent=[-tplot, tplot, 0, nflies], interpolation='nearest')
ax.set_yticks(np.arange(nflies)+.5)
ax.set_yticklabels([f'Fly {fly}' for fly in range(nflies)])
ax.set_xlabel('Frame')
ax.plot([0,0],[0,nflies], '--', lw=1, color=[0.5,0.5,0.5])
ax.set_title('Moving forward')
fig.tight_layout()

outfilestr = os.path.join(figdir, f'behaviorclassification_{expstr}_{f:05d}')
fig.savefig(outfilestr+'.svg')
fig.savefig(outfilestr+'.png', dpi=300)
fig.savefig(outfilestr+'.pdf')

# %% video

expdir = '/groups/branson/bransonlab/datasets/flyopto/VNC2_Line1_RigA_20230621T102329'
expstr = os.path.basename(expdir)
moviefile = os.path.join(expdir, 'movie.ufmf')
movie = movies.Movie(moviefile)
import cv2
t0 = 4622-100
t1 = 4622+500
outfps = 30
outfile = os.path.join(figdir, f'movie_{expstr}_{t0}to{t1}_5xslower.mp4')
downsample = 2

outh = im.shape[0] // downsample
outw = im.shape[1] // downsample

# Create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
out = cv2.VideoWriter(outfile, fourcc, outfps, (outw, outh))

# Write frames
for t in range(t0, t1+1):
    im = movie.get_frame(t)[0]
    im_resized = cv2.resize(im, (outw, outh))
    im_bgr = cv2.cvtColor(im_resized, cv2.COLOR_GRAY2BGR)
    out.write(im_bgr)
    
out.release()

t0 = 4622-100
t1 = 4622+2900

outfile = os.path.join(figdir, f'movie_{expstr}_{t0}to{t1}_RT.mp4')
out = cv2.VideoWriter(outfile, fourcc, outfps, (outw, outh))

# Write frames
for t in range(t0, t1+1, FPS//outfps):
    im = movie.get_frame(t)[0]
    im_resized = cv2.resize(im, (outw, outh))
    im_bgr = cv2.cvtColor(im_resized, cv2.COLOR_GRAY2BGR)
    out.write(im_bgr)
    
out.release()

# %% compute swing/stance durations and velocities

rootdatadir = '/groups/branson/bransonlab/datasets/flyopto'
expdirs = [os.path.join(rootdatadir,d) for d in os.listdir(rootdatadir) if d.startswith('VNC2_')]
nexps = len(expdirs)
# parse line from expdir name
import re
pattern = r'VNC2_(?P<line>.+?)_Rig(?P<rig>[A-Z])_(?P<datetime>\d{8}T\d{6})'
expinfo = []
for expdir in expdirs:
    m = re.match(pattern, os.path.basename(expdir))
    if m:
        expinfo.append(m.groupdict())
    else:
        expinfo.append(None)
swingstancefile = 'swingstance.mat'


# find all stance bouts
def get_interval_ends(x):
    """Get start and end indices of contiguous True intervals in boolean array x."""
    dx = np.diff(x.astype(int))
    starts = np.where(dx == 1)[0] + 1
    ends = np.where(dx == -1)[0] + 1
    if x[0]:
        starts = np.insert(starts, 0, 0)
    if x[-1]:
        ends = np.append(ends, len(x))
    return starts, ends

thresh_speed = np.array([-np.inf,20,np.inf]) # mm/s
#thresh_speed = np.array([-np.inf,20,np.inf]) # mm/s
nbins = len(thresh_speed) - 1

expi = 0
expdir = expdirs[expi]
id,_ = matio.loadmat(os.path.join(expdir,'indicatordata.mat'))
nstimuli = len(id['indicatorLED']['startframe'])

mean_stance_duration_on = np.zeros((nexps,nstimuli,nbins))
n_stance_on = np.zeros((nexps,nstimuli,nbins))
mean_stance_duration_off = np.zeros((nexps,nstimuli,nbins))
n_stance_off = np.zeros((nexps,nstimuli,nbins))

for expi in range(nexps):

    print(f'Processing experiment {expi+1}/{nexps}...')
    mean_stance_duration_on[expi,:,:] = 0
    n_stance_on[expi,:,:] = 0
    mean_stance_duration_off[expi,:,:] = 0
    n_stance_off[expi,:,:] = 0
    
    expdir = expdirs[expi]
    expstr = os.path.basename(expdir)
    swingstancepath = os.path.join(expdir, swingstancefile)
    swingstancedata,_ = matio.loadmat(swingstancepath)
    # load speed data
    speedfile = os.path.join(expdir,'perframe','velmag_ctr.mat')
    speeddata,_ = matio.loadmat(speedfile)
    nflies = swingstancedata['isswing'].shape[2]
    sd,_ = matio.loadmat(os.path.join(expdir,'scores_Walk2.mat'))
    id,_ = matio.loadmat(os.path.join(expdir,'indicatordata.mat'))
    
    for stimi in range(nstimuli):
        stimonstart = int(id['indicatorLED']['startframe'][stimi,0])
        stimonend = int(id['indicatorLED']['endframe'][stimi,0])
        stimoffstart = 0 if stimi==0 else int(id['indicatorLED']['endframe'][stimi-1,0])+1
        stimoffend = int(id['indicatorLED']['startframe'][stimi,0])-1
        for fly in range(nflies):
            firstframe = int(sd['allScores']['tStart'][fly,0]-1)
            endframe = int(sd['allScores']['tEnd'][fly,0]-1)
            # fly must be alive for all frames of stimulus on and off periods
            if firstframe > stimoffstart or endframe < stimonend:
                continue
            for leg in range(6):
                t0s,t1s = get_interval_ends(swingstancedata['isstance'][:,leg,fly]==1)
                idxon = (t1s >= stimonstart) & (t1s <= stimonend)
                t0son = t0s[idxon]
                t1son = t1s[idxon]
                idxoff = (t0s >= stimoffstart) & (t1s <= stimoffend)
                t0soff = t0s[idxoff]
                t1soff = t1s[idxoff]
                stance_durations_on = t1son - t0son
                stance_durations_off = t1soff - t0soff
                stance_speeds_on = np.array([np.mean(speeddata['data'][fly][t0-firstframe:t1-firstframe]) for t0,t1 in zip(t0son,t1son)])
                stance_speeds_off = np.array([np.mean(speeddata['data'][fly][t0-firstframe:t1-firstframe]) for t0,t1 in zip(t0soff,t1soff)])
                for bini in range(nbins):
                    isspeedon = (thresh_speed[bini] <= stance_speeds_on) & (stance_speeds_on < thresh_speed[bini+1])
                    mean_stance_duration_on[expi,stimi,bini] += np.sum(stance_durations_on[isspeedon])
                    n_stance_on[expi,stimi,bini] += np.sum(isspeedon)
                    isspeedoff = (thresh_speed[bini] <= stance_speeds_off) & (stance_speeds_off < thresh_speed[bini+1])
                    mean_stance_duration_off[expi,stimi,bini] += np.sum(stance_durations_off[isspeedoff])
                    n_stance_off[expi,stimi,bini] += np.sum(isspeedoff)
                    
mean_stance_duration_on /= np.maximum(1, n_stance_on)
mean_stance_duration_off /= np.maximum(1, n_stance_off)
mean_stance_duration_on[n_stance_on==0] = np.nan
mean_stance_duration_off[n_stance_off==0] = np.nan

mean_stance_duration_on /= FPS
mean_stance_duration_off /= FPS

# compute difference in on vs off
delta_mean_stance_duration = mean_stance_duration_on - mean_stance_duration_off

linenames,lineidx = np.unique([info['line'] for info in expinfo if info is not None],return_inverse=True)
nlines = len(linenames)
color_control = (0,0,0,1)
colors_lines = plt.cm.tab10.colors
controllineidx = np.where(linenames == 'Control')[0][0]

from scipy import stats

# Get control data
control_data = delta_mean_stance_duration[lineidx == controllineidx, :, :]  # shape (n_control_exp, n_legs, nbins)

# For each non-control line and each bin, compare to control
pvalues = np.zeros((nlines, nbins))
for linei in range(nlines):
    if linei == controllineidx:
        pvalues[linei, :] = np.nan
        continue
    line_data = delta_mean_stance_duration[lineidx == linei, :, :]
    for bini in range(nbins):
        ctrl = control_data[:, :, bini].flatten()
        line = line_data[:, :, bini].flatten()
        # Remove nans
        ctrl = ctrl[~np.isnan(ctrl)]
        line = line[~np.isnan(line)]
        # Mann-Whitney U test (non-parametric) or use stats.ttest_ind for t-test
        _, pvalues[linei, bini] = stats.mannwhitneyu(ctrl, line, alternative='two-sided')

# Print results
for linei in range(nlines):
    if linei == controllineidx:
        continue
    print(f"{linenames[linei]}:")
    for bini in range(nbins):
        sig = '*' if pvalues[linei, bini] < 0.05 else ''
        print(f"  bin {bini}: p={pvalues[linei, bini]:.4f} {sig}")

fig,ax = plt.subplots(1,1, figsize=(2*basefigsize,basefigsize))
ax.cla()
colori = 0
h = []
for linei in range(nlines):
    if linei == controllineidx:
        color = color_control
        off = 0
        zorder = 5
    else:
        color = colors_lines[colori]
        off = -.3+colori*.6/(nlines-1)+.02
        zorder = 2
        colori += 1
    line_expi = np.where(lineidx == linei)[0]
    for bini in range(nbins):
        x = delta_mean_stance_duration[lineidx == linei,:,bini].flatten()
        ax.plot(np.full(x.shape,bini+off), x, '.', color=(color[:3] + (0.5,)), zorder=zorder, markersize=5)
        # plot line from min to max
        sig = np.nanstd(x)
        ax.plot([bini+off, bini+off], [np.nanmean(x)-sig, np.nanmean(x)+sig], '-', color=[c*.5 for c in color[:3]] + [1,], lw=2, zorder=zorder-1)
        #ax.plot([bini+off, bini+off], [np.min(delta_mean_stance_duration[lineidx == linei,bini]), np.max(delta_mean_stance_duration[lineidx == linei,bini])], '-', color=color, lw=1)
    x = delta_mean_stance_duration[lineidx == linei,:,:].reshape(-1, nbins)
    hcurr = ax.plot(np.arange(nbins)+off, np.nanmean(x, axis=0), '-', color=color, lw=2, label=linenames[linei], zorder=zorder+1)[0]
    #hcurr = ax.plot(np.arange(nbins)+off, np.full(nbins, np.nan), '-', color=color, lw=2, label=linenames[linei], zorder=zorder+1)[0]
    h.append(hcurr)
    if linei == controllineidx:
        ax.plot(np.arange(nbins)+off, np.nanmean(x, axis=0), 'o', color=color, lw=2, label=linenames[linei], zorder=zorder+1, markersize=6)
    else:
        for bini in range(nbins):
            if pvalues[linei, bini] < 0.05:
                ax.plot(bini+off, np.nanmean(x[:, bini]), '*', color=color, lw=2, label=linenames[linei], zorder=zorder+2, markeredgecolor='k', markersize=10)
            else:
                ax.plot(bini+off, np.nanmean(x[:, bini]), 'o', color=[0,0,0,0], lw=2, label=linenames[linei], zorder=zorder, markeredgecolor=color, markersize=6)            
            
ax.set_xticks(np.arange(nbins-1)+.5)
ax.set_xticklabels([str(x) for x in thresh_speed[1:-1]])
#ax.set_xticks(np.arange(nbins))
#ax.set_xticklabels(['Slow','Fast'])
#ax.set_xlabel('Speed bin (mm/s)')
ax.set_ylabel('On - off stance duration (s)')
ax.legend(handles=h, loc='upper left', bbox_to_anchor=(1, 1))

ax.set_ylim(-.025,.025)
fig.tight_layout()

outfilestr = 'stance_duration_comparison_alllines'
fig.savefig(os.path.join(figdir, outfilestr+'.svg'))
fig.savefig(os.path.join(figdir, outfilestr+'.png'), dpi=300)
fig.savefig(os.path.join(figdir, outfilestr+'.pdf'))

# %% 
