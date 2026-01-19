tmp = load('/groups/branson/bransonlab/flydisco_data/VNC2_YNA_K_162984_RigA_20230913T100524/locomotionmetrics_swingstancebouts.mat');

nflies = numel(tmp.bout_metrics_ON.perfly);
nlegs = 6;
T = 0;
fns1 = {'bout_metrics_ON','bout_metrics_OFF'};
fns2 = {'swing','stance'};
for fly = 1:nflies,
  for leg = 1:nlegs,
    for i1 = 1:numel(fns1),
      for i2 = 1:numel(fns2),
        T = max(T,max(tmp.(fns1{i1}).perfly(fly).perlimb(leg).(fns2{i2}).end_indices));
      end
    end
  end
end
isswing = false([nflies,nlegs,T]);
isstance = false([nflies,nlegs,T]);
for fly = 1:nflies,
  for leg = 1:nlegs,
    for i1 = 1:numel(fns1),
      fn2 = 'swing';
      t0s = tmp.(fns1{i1}).perfly(fly).perlimb(leg).(fn2).start_indices;
      t1s = tmp.(fns1{i1}).perfly(fly).perlimb(leg).(fn2).end_indices;
      for j = 1:numel(t0s),
        isswing(fly,leg,t0s(j):t1s(j)-1) = true;
      end
      fn2 = 'stance';
      t0s = tmp.(fns1{i1}).perfly(fly).perlimb(leg).(fn2).start_indices;
      t1s = tmp.(fns1{i1}).perfly(fly).perlimb(leg).(fn2).end_indices;
      for j = 1:numel(t0s),
        isstance(fly,leg,t0s(j):t1s(j)-1) = true;
      end
    end
  end
end
iswalk = isswing | isstance;
save('swingstance.mat','isswing','isstance','iswalk');