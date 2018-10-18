function p = prior_pcnu(varargin)
% PRIOR_PCNU  penalized complexity priors structure     
%       
%  Description
%    P = PRIOR_PCNU('PARAM1', VALUE1, 'PARAM2', VALUE2, ...) 
%    creates penalized-complexity prior in which the
%    named parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    P = PRIOR_PCNU(P, 'PARAM1', VALUE1, 'PARAM2', VALUE2, ...)
%    modify a prior structure with the named parameters altered
%    with the specified values.
%
%    The parameterization is on the real line (x) : 1/τ = σ² 
%    
%    Parameters for pc-priors [default]
%      lambda   - penalization parameters [1]
%      ualpha   - vector (U, α)
%
%  See also
%    PRIOR_*
%
%  Penalizing model component complexity: A principled practical approach
%  to constructing priors. Daniel Simpson, Håvard Hue. Thiago G. Martins.
%  Andrea Riebler and Sigrrun Sørbye. ArXiv 2015. Submitted to Statistical Science.
%
% Copyright (c) 2000-2001,2010 Aki Vehtari
% Copyright (c) 2009 Jarno Vanhatalo
% Copyright (c) 2010 Jaakko Riihimäki
% ───────────── 2015 Marcelo Hartmann

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip = inputParser;
  ip.FunctionName = 'PRIOR_PCNU';
  ip.addOptional('p', [], @isstruct);
  ip.addParamValue('lambda', 1, @(x) isscalar(x) && x > 0);
  ip.addParamValue('lambda_prior',[], @(x) isstruct(x) || isempty(x));
  ip.addParamValue('ualpha', [], @(x) isvector(x) && length(x) == 2);
  ip.parse(varargin{:});
  p = ip.Results.p;
  
  if isempty(p)
      init = true;
      p.type = 'pcnu';
  else
      if ~isfield(p, 'type') && ~isequal(p.type, 'pc')
          error('First argument does not seem to be a valid prior structure')
      end
      init = false;
  end

  % initialize 
  if init && ~ismember('ualpha', ip.UsingDefaults)  
      p.ualpha = ip.Results.ualpha;
      
      % solve for λ. P(ν < c) = p
      p.lambda = -log(p.ualpha(2)) .* p.ualpha(1);
      
  elseif init || ~ismember('lambda', ip.UsingDefaults)
      p.lambda = ip.Results.lambda;
      
  end
  
  % Initialize prior structure
  if init
      p.p = [];
  end
  
  if init || ~ismember('lambda_prior', ip.UsingDefaults)
      p.p.lambda = ip.Results.lambda_prior;
  end
  
  if init
    % set functions
    p.fh.pak = @prior_pc_pak;
    p.fh.unpak = @prior_pc_unpak;
    p.fh.lp = @prior_pc_lp;
    p.fh.lpg = @prior_pc_lpg;
    p.fh.recappend = @prior_pc_recappend;
  end

end

function [w, s, h] = prior_pc_pak(p)
% This is a mandatory subfunction used for example 
% in energy and gradient computations.
  
  w = [];
  s = {};
  h = [];
  
  if ~isempty(p.p.lambda)
    w = log(p.lambda);
    s = [s; 'log(pc.lambda)'];
    h = 1;
  end        
  
end

function [p, w] = prior_pc_unpak(p, w)
% This is a mandatory subfunction used for example 
% in energy and gradient computations.

  if ~isempty(p.p.lambda)
      i1 = 1;
      p.lambda = exp(w(i1));
      w = w(i1 + 1:end);
  end
  
end

function lp = prior_pc_lp(x, p)
% This is a mandatory subfunction used for example 
% in energy computations.
  
  lp = log(p.lambda) - 2*log(x) - p.lambda./x;

  if ~isempty(p.p.lambda)
      lp = lp + p.p.lambda.fh.lp(p.lambda, p.p.lambda);
  end

end

function lpg = prior_pc_lpg(x, p)
% This is a mandatory subfunction used for example 
% in gradient computations.

  lpg = -2./x + p.lambda./(x.^2);
    
  if ~isempty(p.p.lambda)
      lpglambda = -sqrt(x) + 1/p.lambda + p.p.lambda.fh.lpg(p.lambda, p.p.lambda);
      lpg = [lpg lpglambda];
  end
  
end

function rec = prior_pc_recappend(rec, ri, p)
% This subfunction is needed when using MCMC sampling (gp_mc).

% The parameters are not sampled in any case.
rec = rec;
if ~isempty(p.p.lambda)
    rec.mu(ri, :) = p.lambda;
end        

end
