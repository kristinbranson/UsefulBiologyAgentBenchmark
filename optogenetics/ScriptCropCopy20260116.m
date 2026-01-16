% crop registered_trx.mat

expinfofile = '/groups/branson/home/bransonk/behavioranalysis/code/ScienceBenchmark/optogenetics/expdirlist20260115_anonymized.csv';
expinfo = importdata(expinfofile);

expdirsin = expinfo.textdata(:,1);
expdirsout = expinfo.textdata(:,2);
endframes = expinfo.data(:,1);
nexps = numel(expdirsin);
maxoff = 1;
tfforce = false;

%% registered trx file

filestr = 'registered_trx.mat';
%expi = 1;
for expi = 1:nexps,
  expdirin = expdirsin{expi};
  expdirout = expdirsout{expi};
  endframe = endframes(expi);
  infile = fullfile(expdirin,filestr);
  outfile = fullfile(expdirout,filestr);
  if (~tfforce) && exist(outfile,'file'),
    fprintf('Skipping %s, already done\n',outfile);
    continue;
  end
  crop_trx_file(infile,outfile,endframe);
end

%% wing tracking trx file

filestr = 'wingtracking_results.mat';
for expi = 1:nexps,
  expdirin = expdirsin{expi};
  expdirout = expdirsout{expi};
  endframe = endframes(expi);
  infile = fullfile(expdirin,filestr);
  outfile = fullfile(expdirout,filestr);
  if (~tfforce) && exist(outfile,'file'),
    fprintf('Skipping %s, already done\n',outfile);
    continue;
  end
  crop_trx_file(infile,outfile,endframe,'maxoff',maxoff,'tfforce',tfforce);
end

%% copy over sexclassifier.mat

filestr = 'sexclassifier.mat';
for expi = 1:nexps,
  fprintf('%d/%d\n',expi,nexps);
  expdirin = expdirsin{expi};
  expdirout = expdirsout{expi};
  endframe = endframes(expi);
  infile = fullfile(expdirin,filestr);
  outfile = fullfile(expdirout,filestr);
  if (~tfforce) && exist(outfile,'file'),
    fprintf('Skipping %s, already done\n',outfile);
    continue;
  end
  copyfile(infile,outfile);
end

%% crop indicator data

for expi = 1:nexps,
  fprintf('%d/%d\n',expi,nexps);
  expdirin = expdirsin{expi};
  expdirout = expdirsout{expi};
  endframe = endframes(expi);
  infile = fullfile(expdirin,filestr);
  inmoviefile = fullfile(expdirin,'movie.ufmf');
  outfile = fullfile(expdirout,filestr);
  if (~tfforce) && exist(outfile,'file'),
    fprintf('Skipping %s, already done\n',outfile);
    continue;
  end
  idin = load(infile);
  [~,~,fid,headerinfo] = get_readframe_fcn(inmoviefile);
  fclose(fid);
  timestamps = headerinfo.timestamps;

  idout = idin;
  idout.indicatorLED.indicatordigital = idin.indicatorLED.indicatordigital(1:endframe);
  [idout.indicatorLED.startframe,idout.indicatorLED.endframe] = get_interval_ends(idout.indicatorLED.indicatordigital);
  idout.indicatorLED.starttimes = timestamps(idout.indicatorLED.startframe);
  idout.indicatorLED.endtimes = timestamps(idout.indicatorLED.startframe);
  save(outfile,'-struct','idout');
end

%% scores files

filestr = 'scores_Walk2.mat';
for expi = 1:nexps,
  fprintf('%d/%d\n',expi,nexps);

  expdirin = expdirsin{expi};
  expdirout = expdirsout{expi};
  endframe = endframes(expi);
  infile = fullfile(expdirin,filestr);
  outfile = fullfile(expdirout,filestr);
  if (~tfforce) && exist(outfile,'file'),
    fprintf('Skipping %s, already done\n',outfile);
    continue;
  end
  crop_score_file(infile,outfile,endframe);
end

%% per-frame data
perframedir = 'perframe';
for expi = 1:nexps,
  fprintf('%d/%d\n',expi,nexps);
  expdirin = expdirsin{expi};
  expdirout = expdirsout{expi};
  endframe = endframes(expi);
  indir = fullfile(expdirin,perframedir);
  outdir = fullfile(expdirout,perframedir);
  tdin = load(fullfile(expdirin,'registered_trx.mat'));
  if ~exist(outdir,'dir'),
    mkdir(outdir);
  end
  inperframefiles = mydir(fullfile(indir,'*.mat'));
  for i = 1:numel(inperframefiles),
    [~,filestr,ext] = fileparts(inperframefiles{i});
    infile = inperframefiles{i};
    outfile = fullfile(outdir,[filestr,ext]);
    if ~tfforce && exist(outfile,'file'),
      continue;
    end
    inpd = load(infile);
    outpd = inpd;
    dokeep = true(1,numel(inpd.data));
    for fly = 1:numel(inpd.data),
      if tdin.trx(fly).firstframe > endframe,
        dokeep(fly) = false;
        continue;
      end
      if tdin.trx(fly).endframe <= endframe,
        continue;
      end
      j1 = endframe - tdin.trx(fly).firstframe + 1;
      off = tdin.trx(fly).nframes - numel(inpd.data{fly});
      outpd.data{fly} = inpd.data{fly}(1:j1-off);
    end
    save(outfile,'-struct','outpd');
  end
end