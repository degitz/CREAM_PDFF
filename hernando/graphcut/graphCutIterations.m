% Function: graphCutIterations
%
% Description: graph cut iterations for field map estimation
% 
% Parameters:
% Input: structures imDataParams and algoParams
%   - imDataParams.images: acquired images, array of size[nx,ny,1,ncoils,nTE]
%   - imDataParams.TEs: echo times (in seconds)
%   - imDataParams.fieldStrength: (in Tesla)
%
%   - algoParams.species(ii).name = name of species ii (string)
%   - algoParams.species(ii).frequency = frequency shift in ppm of each peak within species ii
%   - algoParams.species(ii).relAmps = relative amplitude (sum normalized to 1) of each peak within species ii
%   Example
%      - algoParams.species(1).name = 'water' % Water
%      - algoParams.species(1).frequency = [0] 
%      - algoParams.species(1).relAmps = [1]   
%      - algoParams.species(2).name = 'fat' % Fat
%      - algoParams.species(2).frequency = [3.80, 3.40, 2.60, 1.94, 0.39, -0.60]
%      - algoParams.species(2).relAmps = [0.087 0.693 0.128 0.004 0.039 0.048]
% 
%   - algoParams.size_clique = 1; % Size of MRF neighborhood (1 uses an 8-neighborhood, common in 2D)
%   - algoParams.range_fm = [-400 400]; % Range of field map values
%   - algoParams.NUM_FMS = 301; % Number of field map values to discretize
%   - algoParams.MAX_ITERS = 40; % Number of graph cut iterations
%   - algoParams.lambda = 0.05; % Regularization parameter
%   - algoParams.residual: in case we pre-computed the fit residual (mostly for testing) 
%
%   - residual: the fit residual, of size NUM_FMS X sx X sy 
%   - lmap: spatially varying regularization parameter
%   - cur_ind: initial indices for field map (the indices determine the field map)
%
% Returns: 
%  - fm: field map
%
% Author: Diego Hernando
% Date created: June 3, 2009
% Date last modified: August 18, 2011

function [fm,index_fm] = graphCutIterations(imDataParams,algoParams,residual,lmap,cur_ind,PTYPE,DEBUG)

if nargin < 7
    DEBUG = 0;
end

if isfield(algoParams, 'seed')
    rng(algoParams.seed)
end

% Initialize primary variables
SMOOTH_NOSIGNAL = algoParams.SMOOTH_NOSIGNAL;
STARTBIG = algoParams.STARTBIG;
dkg = algoParams.dkg;
plot_debug = algoParams.plot_debug;
MIN_ITERS = algoParams.MIN_ITERS;
convTol = algoParams.convTol;
MAX_STALLEDITERS = algoParams.MAX_STALLEDITERS;

% Initialize some auxiliary variables
deltaF = [0; algoParams.gyro*(algoParams.species(2).frequency(:)-algoParams.species(1).frequency(1))*(imDataParams.FieldStrength)];
lambda = algoParams.lambda;
[sx,sy,N,C,num_acqs] = size(imDataParams.images);
fms = linspace(algoParams.range_fm(1),algoParams.range_fm(2),algoParams.NUM_FMS);
dfm = fms(2)-fms(1);
resoffset = (0:(sx*sy-1))'*algoParams.NUM_FMS;
[masksignal,resLocalMinima,numMinimaPerVoxel] = findLocalMinima(residual, 0.06);
numLocalMin = size(resLocalMinima,1);
stepoffset = (0:(sx*sy-1))'*numLocalMin;
ercurrent = 1e10;
clear cur_ind2

if strcmp(PTYPE,'Bipolar_GC')
    dt = imDataParams.TE(floor(size(imDataParams.images,5)/2) + 1) - imDataParams.TE(1);
    negsignVal = 1;
else % Default hernando
    dt = imDataParams.TE(2)-imDataParams.TE(1);
    negsignVal = 20;
end
period = 1/dt;

% Main graph-cut loop. At each iteration, a max-flow/min-cut problem is solved
erPrev = sum(residual(cur_ind(:)+resoffset(:)),'all') + dfm^2*lambda*lmap(1,1)*(sum(abs(diff(cur_ind,2,1)).^2,'all') + sum(abs(diff(cur_ind,2,2)).^2,'all'));
fm = zeros(sx,sy);
fmiters = zeros(sx,sy,algoParams.MAX_ITERS);
kg = 0;
noImprove  = 0;
while kg < algoParams.MAX_ITERS
    kg = kg + 1;
    fmiters(:,:,kg) = fm;

    if kg == 1 && STARTBIG
        lambdamap = lambda*lmap;
        ercurrent = 1e10;
        prob_bigJump = 1;
    elseif (kg == dkg && SMOOTH_NOSIGNAL) || ~STARTBIG
        lambdamap = lambda*lmap;
        ercurrent = 1e10;
        prob_bigJump = 0.5;
    end

    % Get the sign of the current jump
    cur_sign = (-1)^(kg);

    % Configure the current jump move
    if rand < prob_bigJump % If we're making a "jumpmin" move.
        cur_ind2(1,:,:) = cur_ind;
        repCurInd = repmat(cur_ind2,[numLocalMin,1,1]);
        if cur_sign > 0
            stepLocator = (repCurInd+20/dfm>=resLocalMinima) & (resLocalMinima>0);
            stepLocator = squeeze(sum(stepLocator,1))+1;
            validStep = masksignal>0 & stepLocator<=numMinimaPerVoxel;
        else
            stepLocator = (repCurInd-negsignVal/dfm>resLocalMinima) & (resLocalMinima>0);
            stepLocator = squeeze(sum(stepLocator,1));
            validStep = masksignal>0 & stepLocator>=1;
        end
        nextValue = zeros(sx,sy);
        nextValue(validStep) = resLocalMinima(stepoffset(validStep) + stepLocator(validStep));
        cur_step = zeros(sx,sy);
        cur_step(validStep) = nextValue(validStep) - cur_ind(validStep);

        if rand < 0.5
            nosignal_jump = cur_sign*round(abs(deltaF(2))/dfm);
        else
            nosignal_jump = cur_sign*abs(round((period - abs(deltaF(2)))/dfm));
        end
        cur_step(~validStep) = nosignal_jump;

    else % If we're making a standard jump move.
        all_jump = cur_sign*ceil(abs(randn*3));
        cur_step = all_jump*ones(sx,sy);
        nextValue = cur_ind + cur_step;

        % DH* 100309: fix errors where fm goes beyond range
        if cur_sign > 0
            cur_step(nextValue(:)>algoParams.NUM_FMS) = algoParams.NUM_FMS - cur_ind(nextValue(:)>algoParams.NUM_FMS);
        else
            cur_step(nextValue(:)<1) = 1 - cur_ind(nextValue(:)<1);
        end

        if DEBUG
            all_jump %#ok<NOPRT>
        end
    end

    % Create the huge adjacency matrix
    % if norm(lambdamap,'fro') > 0
        [A] = createExpansionGraphVARPRO_fast( residual, dfm, lambdamap, algoParams.size_clique,cur_ind, cur_step);
    % else
    %     [A] = createExpansionGraphVARPRO_fast( residual, dfm, lambdamap, algoParams.size_clique,cur_ind, cur_step);
    % end
    A(A<0) = 0;

    % Solve the max-flow/min-cut problem (max_flow is a function of the matlabBGL library)
    [flowvalTS,cut_TS,RTS,FTS] = max_flow(A',size(A,1),1);

    % Take the output of max_flow and update the fieldmap estimate with
    % the best neighbor (among a set of 2^Q neighbors, where Q is the
    % number of voxels)
    cut1 = (cut_TS==-1);
    cut1b = 0*cut1 + 1;
    cut1b(end) = 0;

    if DEBUG
        compareGC = [sum(sum(A(cut1b==1, cut1b==0))),  sum(sum(A(cut1==1, cut1==0)))] %#ok<NOPRT>
    end

    if sum(sum(A(cut1b==1, cut1b==0))) <= sum(sum(A(cut1==1, cut1==0)))
        cur_indST = cur_ind;
        erST = erPrev;                     % rejected move -> energy unchanged
        if DEBUG, disp('Not taken'); end
    else
        cut = reshape(cut1(2:end-1)==0,sx,sy);
        cur_indST = cur_ind + cur_step.*(cut);
        erST = sum(residual(cur_indST(:)+resoffset(:)),'all') + ...
            dfm^2*lambdamap(1,1)*(sum(abs(diff(cur_indST,2,1)).^2,'all') + sum(abs(diff(cur_indST,2,2)).^2,'all'));
        if DEBUG, disp('Taken'); end
    end

    % Update the field map (described as a map of indices in this function)
    prev_ind = cur_ind;
    cur_ind = cur_indST;

    cur_ind(cur_ind<1) = 1;
    cur_ind(cur_ind>algoParams.NUM_FMS) = algoParams.NUM_FMS;

    fm = fms(cur_ind);

    index_fm = cur_ind;
  
    % Check for convergence
    relImprove = abs(erPrev - erST) / (abs(erPrev) + eps);
    if kg > MIN_ITERS && relImprove < convTol
        noImprove = noImprove + 1;
    else
        noImprove = 0;
    end
    erPrev = erST;

    if noImprove >= MAX_STALLEDITERS
        if DEBUG
            fprintf('Converged at iter %d; energy plateaued for %d iters\n', kg, MAX_STALLEDITERS);
        end
        break
    end

    if plot_debug
        fprintf('%i, %.4e, %i\n', kg, relImprove, noImprove)
        % If we want to see the fieldmap at each step
        if strcmp(PTYPE,'Bipolar_GC')
            figure(100)
        else
            figure
        end
        imagesc(fm)
        axis image
        axis off
        % clim(algoParams.range_fm)
        title(['Iteration: ' num2str(kg)],'FontSize',12);
        colormap gray
        colorbar
        try
            %[kg fm(75,65) fms(cur_ind(75,65)) fms(cur_ind(75,65) + cur_step(75,65)) cur_ind(75,65) cur_step(75,65)]
        end
        drawnow;
    end
end