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
rootdatadir = '/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/data'
expname = 'socialCsChr_JRC_SS57589_CsChrimson_RigC_20190910T165024'
moviefilestr = 'movie.ufmf'
flytrackerfilestr = 'registered_trx.mat'
indicatorfilestr = 'indicatordata.mat'
aptfilestr = 'apttrk.mat'

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
ax.imshow(im, cmap='gray')
ax.set_title(f'Frame {f}, timestamp {timestamp:.3f} s')
movieobj.close()

# %%
# load flytracker data
flytrackerfile = os.path.join(expdir, flytrackerfilestr)
assert os.path.exists(flytrackerfile), f'Flytracker file {flytrackerfile} does not exist'
print(f'Loading flytracker data from {flytrackerfile}')
trx = TrkFile.load_trx(flytrackerfile)
