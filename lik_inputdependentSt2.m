function lik = lik_inputdependentSt2(varargin)
%LIK_T  Create a Student-t likelihood structure 
%
%  Description
%    LIK = LIK_T('PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    creates Student-t likelihood structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    LIK = LIK_T(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    modify a likelihood structure with the named parameters
%    altered with the specified values.
%  
%    Parameters for Student-t likelihood [default]
%      sig2         - scale [1]
%      nu           - degrees of freedom [4]
%      sig2_prior   - prior for sigma2 [prior_logunif]
%      nu_prior     - prior for nu [prior_fixed]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    The likelihood funciton for many observation is defined as follows:
%                         __ n
%    L(y|f1, f2, σ2, ν) = || i=1 C(σ, ν) *
%                         (1 + 1/ν * ((y_i - f_1)/σ)^2 )^(-(ν + 1)/2)
%
%    where, 
%   
%    σ = σ2 exp(f2)
%    C(σ, ν) = Γ((ν + 1)/2) / (Γ(ν/2) σ sqrt(πν))
%
%    Ιn this case σ2 (mean in the log scale for the processes
%    f2) is the scale parameters and nu is the degrees of freedom. 
%    Note that we are modelling the scale parameter in the Student-t 
%    probability model as GP models.
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2011 Pasi Jylänki
% ───────────── 2016 Marcelo Hartmann

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.
%LIK_T  Create a Student-t likelihood structure 

  ip = inputParser;
  ip.FunctionName = 'LIK_ST';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('sig2', 1, @(x) isscalar(x) && x > 0);
  ip.addParamValue('sig2_prior', prior_t(), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('nu', 4, @(x) isscalar(x) && x > 0);
  ip.addParamValue('nu_prior', prior_invt('s2', 0.25), @(x) isstruct(x) || isempty(x));
  ip.addParamValue('fisher', 'off', @(x) strcmp(x, 'on') || strcmp(x, 'off'));
 
  ip.parse(varargin{:});
  lik=ip.Results.lik;
  
  if isempty(lik)
      init = true;
      lik.nondiagW = true;
      lik.type = 'inputdependentt';
  else
      if ~isfield(lik,'type') || ~isequal(lik.type, 'inputdependentt')
          error('First argument does not seem to be a valid likelihood function structure');
      end
      init = false;
  end
  % Initialize fisher option
  if init || ~ismember('fisher', ip.UsingDefaults)
      lik.fisher = ip.Results.fisher;
  end
  
  % Initialize parameters
  if init || ~ismember('sig2', ip.UsingDefaults)
      lik.sig2 = ip.Results.sig2;
  end
  if init || ~ismember('nu', ip.UsingDefaults)
      lik.nu = ip.Results.nu;
  end
  
  % Initialize prior structure
  if init
      lik.p = [];
  end
  if init || ~ismember('sig2_prior', ip.UsingDefaults)
      lik.p.sig2 = ip.Results.sig2_prior;
  end
  if init || ~ismember('nu_prior', ip.UsingDefaults)
      lik.p.nu = ip.Results.nu_prior;
  end
    
  % Set the function handles to the subfunctions
  if init
      lik.fh.pak = @lik_inputdependentT_pak;
      lik.fh.unpak = @lik_inputdependentT_unpak;
      lik.fh.lp = @lik_inputdependentT_lp;
      lik.fh.lpg = @lik_inputdependentT_lpg;
      lik.fh.ll = @lik_inputdependentT_ll;
      lik.fh.llg = @lik_inputdependentT_llg;    
      lik.fh.llg2 = @lik_inputdependentT_llg2;
      lik.fh.fi = @lik_inputdependentT_fi;
      lik.fh.llg3 = @lik_inputdependentT_llg3;
      lik.fh.dfi = @lik_inputdependentT_dfi;
      lik.fh.invlink = @lik_inputdependentT_invlink;
      lik.fh.predy = @lik_inputdependentT_predy;
      lik.fh.predprcty = @lik_inputdependentT_predprcty;
      lik.fh.recappend = @lik_inputdependentT_recappend;
  end
end

function [w, s, h] = lik_inputdependentT_pak(lik)
%LIK_INPUTDEPENDENTT_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_INPUTDEPENDENTT_PAK(LIK) takes a likelihood structure LIK and
%    combines the parameters into a single row vector W. This 
%    is a mandatory subfunction used for example in energy and 
%    gradient computations.
%     
%       w = [ log(lik.sig2)
%             (hyperparameters of lik.sig2)
%             log(lik.nu)
%             (hyperparameters of lik.nu) ]'
%
%  See also
%    LIK_T_UNPAK, GP_PAK
  
  w = []; s = {}; h = [];
  
  if ~isempty(lik.p.sig2)
      w = [w log(lik.sig2)];
      s = [s; 'log(lik.sig2)'];
      h = [h 0];
      
      [wh, sh, hh] = lik.p.sig2.fh.pak(lik.p.sig2);
      w = [w wh];
      s = [s; sh];
      h = [h hh];
  end
  
  if ~isempty(lik.p.nu)
      w = [w log(lik.nu)];
      s = [s; 'log(lik.nu)'];
      h = [h 0];
      
      [wh, sh, hh] = lik.p.nu.fh.pak(lik.p.nu);
      w = [w wh];
      s = [s; sh];
      h = [h hh];
  end        
end

function [lik, w] = lik_inputdependentT_unpak(lik, w)
%LIK_INPUTDEPENDENTT_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_INPUTDEPENDENTT_UNPAK(W, LIK) takes a likelihood structure LIK and
%    extracts the parameters from the vector W to the LIK
%    structure. This is a mandatory subfunction used for example 
%    in energy and gradient computations.
%     
%    Assignment is inverse of  
%       w = [ log(lik.sig2)
%             (hyperparameters of lik.sig2)
%             log(lik.nu)
%             (hyperparameters of lik.nu)]'
%
%   See also
%   LIK_T_PAK, GP_UNPAK

  if ~isempty(lik.p.sig2)
        lik.sig2 = exp(w(1));
               w = w(2:end);
          [p, w] = lik.p.sig2.fh.unpak(lik.p.sig2, w);
      lik.p.sig2 = p;
  end
  
  if ~isempty(lik.p.nu) 
        lik.nu = exp(w(1));
             w = w(2:end);
        [p, w] = lik.p.nu.fh.unpak(lik.p.nu, w);
      lik.p.nu = p;
  end
end

function lp = lik_inputdependentT_lp(lik)
%LIK_INPUTDEPENDENTT_LP  log(prior) of the likelihood parameters
%
%  Description
%    LP = LIK_INPUTDEPENDENTT_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters.
%    This subfunction is needed when there are likelihood parameters.
%
%  See also
%    LIK_INPUTDEPENDENTT_LLG, LIK_INPUTDEPENDENTT_LLG3,
%    LIK_INPUTDEPENDENTT_LLG2, GPLA_E
  
  sig2 = lik.sig2;
  nu = lik.nu;
  
  lp = 0;
  
  if ~isempty(lik.p.sig2) 
      lp = lp + lik.p.sig2.fh.lp(sig2, lik.p.sig2) + log(sig2);
  end
  
  if ~isempty(lik.p.nu)
      lp = lp + lik.p.nu.fh.lp(lik.nu, lik.p.nu) + log(nu);
  end
end

function lpg = lik_inputdependentT_lpg(lik)
%LIK_INPUTDEPENDENTT_LPG  d log(prior)/dth of the likelihood parameters th
%
%  Description
%    LPG = LIK_INPUTDEPENDENTT_LPG(LIK) takes a likelihood structure LIK
%    and returns d log(p(th))/dth, where th collects the
%    parameters. This subfunction is needed when there are 
%    likelihood parameters.
%
%  See also
%    LIK_INPUTDEPENDENTT_LLG, LIK_INPUTDEPENDENTT_LLG3,
%    LIK__INPUTDEPENDENTTLLG2, GPLA_G
  
% Evaluate the gradients of log(prior)

  sig2 = lik.sig2;
  nu = lik.nu;
  
  lpg = [];
  i1 = 0;
  
  if ~isempty(lik.p.sig2) 
      i1 = i1 + 1;
      lpg(i1) = lik.p.sig2.fh.lpg(lik.sig2, lik.p.sig2) .* sig2 + 1;
  end
  if ~isempty(lik.p.nu)
      i1 = i1 + 1;
      lpg(i1) = lik.p.nu.fh.lpg(lik.nu, lik.p.nu) .* nu + 1;
  end
end

function ll = lik_inputdependentT_ll(lik, y, f, z)
%LIK_INPUTDENPENDENTT_LL  Log likelihood
%
%  Description
%    LL = LIK_INPUTDENPENDENTT_LL(LIK, Y, F) takes a likelihood structure LIK,
%    observations Y, and latent values F. Returns the log
%    likelihood, log p(y|f,z). This subfunction is needed when 
%    using Laplace approximation or MCMC for inference with 
%    non-Gaussian likelihoods. This subfunction is also used in
%    information criteria (DIC, WAIC) computations.
%
%  See also
%    LIK_INPUTDENPENDENTT_LLG, LIK_INPUTDENPENDENTT_LLG3, 
%    LIK_INPUTDENPENDENTT_LLG2, GPLA_E

  n = size(y, 1);
  
  sig2 = lik.sig2;
  nu   = lik.nu;
  
  f  = f(:);
  f1 = f(1:n); 
  f2 = f(n+1 : 2*n);
  
  sig = sig2 .* exp(f2);
  sig(sig < eps) = eps;
  sig(sig > 1e6) = 1e6;
  
  z  = (y - f1) ./ sig;
  zA = z.^2;
  zB = 1 + 1/nu .* zA;
  
  ll = sum(gammaln((nu + 1) / 2) - gammaln(nu/2) - log(sig) ... 
        - log(sqrt(pi*nu)) - (nu + 1)/2 .* log(zB));
end


function llg = lik_inputdependentT_llg(lik, y, f, param, z)
%LIK_INPUTDEPENDENTT_LLG  Gradient of the log likelihood
%
%  Description
%    LOGLIKG = LIK_INPUTDEPENDENTT_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y, and latent values F. Returns
%    the gradient of log likelihood with respect to PARAM. At the
%    moment PARAM can be 'param' or 'latent'. This subfunction is 
%    needed when using Laplace approximation or MCMC for inference 
%    with non-Gaussian likelihoods.
%
%  See also
%    LIK_INPUTDEPENDENTT_LL, LIK_INPUTDEPENDENTT_LLG2,
%    LIK_INPUTDEPENDENTT_LLG3, GPLA_E
  
  n = size(y, 1);
  
  sig2 = lik.sig2;
  nu   = lik.nu;
  
  f  = f(:);
  f1 = f(1:n); 
  f2 = f(n+1 : 2*n);
  
  sig = sig2 .* exp(f2);
  sig(sig < eps) = eps;
  sig(sig > 1e6) = 1e6;
  
  z  = (y - f1) ./ sig;
  zA = z.^2;
  zB = 1 + 1/nu .* zA;
  
  switch param
      case 'latent'
          % llg = [dll/df1, dll/df2];
          llg = [ (1 + 1/nu).*z./(sig .* zB);
                  (zA - 1)./zB ];
      
      case 'param'
          i1 = 0;
          
          if ~isempty(lik.p.sig2)
              i1 = i1 + 1;
              % Derivative w.r.t. log(sig2)
              llg(i1) = sum((zA - 1)./zB);

          end
          
          if ~isempty(lik.p.nu)
              i1 = i1 + 1;
              % Derivative w.r.t. to log(nu)
              llg(i1) = 0.5 * sum(psi((nu + 1)./2) - psi(nu./2) - 1./nu ...
                        - log(1 + 1/nu .* zA) + (nu + 1) .* 1/nu^2 .* zA./zB) * nu;
          end
          
  end
  
end


function [llg2, llg22] = lik_inputdependentT_llg2(lik, y, f, param, z)
%LIK_INPUTDEPENDENTT_LLG2  Second gradients of log likelihood
%
%  Description        
%    LLG2 = LIK_INPUTDEPENDENT_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y, and latent values F. Returns
%    the Hessian of log likelihood with respect to PARAM. At the
%    moment PARAM can be only 'latent'. LLG2 is a vector with
%    diagonal elements of the Hessian matrix (off diagonals are
%    zero). This subfunction is needed when using Laplace 
%    approximation or EP for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_INPUTDEPENDENT_LL, LIK_INPUTDEPENDENT_LLG, LIK_INPUTDEPENDENT_LLG3, GPLA_E

  n = size(y, 1);
  
  sig2 = lik.sig2;
  nu   = lik.nu;
  
  f  = f(:);
  f1 = f(1:n); 
  f2 = f(n+1 : 2*n);
  
  sig = sig2 .* exp(f2);
  sig(sig < eps) = eps;
  sig(sig > 1e6) = 1e6;
  
  z  = (y - f1) ./ sig;
  zA = z.^2;
  zB = 1 + 1/nu .* zA;
  
  llg2 = [];

  switch param
    case 'param'
        % ----- %
      
    case 'latent'  % hessian 
        % d²l/df1²
        d2f1 = -1./(sig.^2) .* (1 + 1/nu) .* (2./zB.^2 - 1./zB);
        
        % d²l/df1 df2 (sig on the log scale)
        d2f1f2 = -2./sig .* (1 + 1/nu) .* z ./ zB.^2;
        
        % d²l/df2² = d²sig (df2)² + dsig d²f2 (on the log-scale)
        %d2f2 = -(-1 + (1 + 1/nu).*zA./zB .* (1 + 2./zB)) + (zA - 1)./zB;
        d2f2 = -(2/nu + 2).*zA./(zB.^2);
        
        llg2 = [d2f1, d2f1f2; d2f1f2, d2f2];
        llg2(isinf(llg2)) = sign(llg2(isinf(llg2))) .* realmax;
        
        if nargout > 1
            llg22 = [diag(d2f1), diag(d2f1f2); diag(d2f1f2), diag(d2f2)];
            llg22(isinf(llg22)) = sign(llg22(isinf(llg22))) .* realmax;
        end
        
    case 'latent+param'
        if ~isempty(lik.p.sig2)
            % d²ll/df1 dsig2 (on the log scale)
            d2df1s = -2./sig .* (1 + 1/nu) .* z ./ zB.^2;
            
            % d²ll/df2 dsig2 (on the log scale) 
            d2df2s = -(-1 + (1 + 1/nu).*zA./zB .* (1 + 2./zB)) + (zA - 1)./zB;
            
            llg2 = [d2df1s; d2df2s];
        end
        
        if ~isempty(lik.p.nu)
            % d²ll/df1 dnu (on the log-scale)
            d2f1nu = 1./(sig * nu) .* z .* (zA - 1) ./ zB.^2;
            
            % d²ll/df2 dnu (on the log-scale)
            d2f2nu = 1./nu * (zA - 1) .* zA ./ zB.^2;
            
            llg2 = [llg2, [d2f1nu; d2f2nu]];
        end
  end
end

function llg3 = lik_inputdependentT_llg3(lik, y, f, param, z)
%LIK_INPUTDEPENDENT_LLG3  Third gradients of log likelihood (energy)
%
%  Description
%    LLG3 = LIK_INPUTDEPENDENT_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y and latent values F and
%    returns the third gradients of log likelihood with respect
%    to PARAM. At the moment PARAM can be only 'latent'. G3 is a
%    vector with third gradients. This subfunction is needed when 
%    using Laplace approximation for inference with non-Gaussian 
%    likelihoods.
%
%  See also
%    LIK_INPUTDEPENDENT_LL, LIK_INPUTDEPENDENT_LLG, LIK_INPUTDEPENDENT_LLG2,
%    GPLA_E, GPLA_G

  n = size(y, 1);
  
  sig2 = lik.sig2;
  nu   = lik.nu;
  
  f  = f(:);
  f1 = f(1:n); 
  f2 = f(n+1 : 2*n);
  
  sig = sig2 .* exp(f2);
  sig(sig < eps) = eps;
  sig(sig > 1e6) = 1e6;
  
  z  = (y - f1) ./ sig;
  zA = z.^2;
  zB = 1 + 1/nu .* zA;
  
  switch param
    case 'param'
        % ----- %
      
    case 'latent'
        % - derivative of -W w.r.t. f1 - %
        % d³l/df1 df1 df1
        df111 = -(1./sig.^3) .* (1 + 1/nu) .* (2*z ./ (nu*zB.^2) .* (4./zB - 1));
        
        % d³l/df1 df1 df2 (on the log-scale)
        df121 = (2./sig.^2) .* (1 + 1/nu)./(zB.^2) .* (1 - 4*zA./(nu*zB));
        
        % d³l/df2 df2 df1 (on the log-scale)
        df221 = (1./sig) .* (1 + 1/nu).*z./(zB.^2) .* (4 - 8.*zA./(nu .* zB));
        
        % df221 = (1./sig) .* (1 + 1/nu).*z./(zB.^2) .* (6 - 8.*zA./(nu .* zB)) ...
        %        -2./sig .* (1 + 1/nu) .*z./ zB.^2;
        
            
        % - derivative of -W w.r.t. f2 - %
        % d³l/df1 df1 df2
        df112 = df121;
        
        % d³l/df1 df2 df2 
        df122 = df221;
        
        % d³l/df2 df2 df2 = (d³l/dsig³)*(sig³) + 3*(d²l/dsig²)*(sig²) + (dl/dsig)*(sig)
        df222 = -2 + 2*(1 + 1/nu)./zB .* (2*zA + (4*zA - (zA.^2)/nu)./zB - 4*(zA.^2)./ (nu*zB.^2)) ...
                - 3 * (-1 + (1 + 1/nu).*zA./zB .* (1 + 2./zB)) + (zA - 1)./zB;
            
        % rearranging the third derivatives for gpla_g.
        llg3 = zeros(2, 2, 2, n);
        
        % - derivative of -W w.r.t. f1 - %
        llg3(1, 1, 1, :) = df111;
        llg3(1, 2, 1, :) = df121;
        llg3(2, 1, 1, :) = llg3(1, 2, 1, :);
        llg3(2, 2, 1, :) = df221;
        
        
        % - derivative of -W w.r.t. f2 - %
        llg3(1, 1, 2, :) = df112;   
        llg3(1, 2, 2, :) = df122;
        llg3(2, 1, 2, :) = llg3(1, 2, 2, :);
        llg3(2, 2, 2, :) = df222;   
        
    case 'latent2+param'
        llg3 = [];
        
        if ~isempty(lik.p.sig2)
            % - third derivatives of -W w.r.t log(sigma2) - %
            % d³l/df1 df1 dlog(sigma2)
            df11s = (2./sig.^2) .* (1 + 1/nu)./(zB.^2) .* (1 - 4*zA./(nu*zB));
            
            % d³l/df1 df2 dlog(sigma2)
            df12s = (1./sig) .* (1 + 1/nu).*z./(zB.^2) .* (4 - 8.*zA./(nu .* zB));
            % df12s = (1./sig) .* (1 + 1/nu).*z./(zB.^2) .* (6 - 8.*zA./(nu .* zB)) ...
            %    -2./sig .* (1 + 1/nu) .* z ./ zB.^2;
            
            % d³l/df1 df2 dlog(sigma2)
            df22s = -2 + 2*(1 + 1/nu)./zB .* (2*zA + (4*zA - (zA.^2)/nu)./zB - 4*(zA.^2)./ (nu*zB.^2)) ...
                - 3 * (-1 + (1 + 1/nu).*zA./zB .* (1 + 2./zB)) + (zA - 1)./zB;
            
            llg3 = [diag(df11s), diag(df12s); diag(df12s), diag(df22s)];
        end
        
        if ~isempty(lik.p.nu)
            % - third derivatives of -W w.r.t log(nu) - %
            % d³l/df1 df1 dlog(nu)
            df11nu = -1./(sig.^2) .* (-1/(nu^2) * (2./(zB.^2) - 1./zB) + ...
                (1 + 1/nu) * (4*zA./(nu^2 * zB.^3) - zA./(nu^2 * zB.^2))) .* nu;
            
            % d³l/df1 df2 dlog(nu)
            df12nu = 2./(sig.* zB.^2) .* (z./nu^2 - 2*(1 + 1/nu)*z.^3./(nu^2 * zB)) .* nu;
            
            % d³l/df2 df2 dlog(nu)
            df22nu = -(-zA./(nu^2).*(2./(zB.^2) + 1./zB) + ...
                (1 + 1/nu).*((zA.^2)./(nu^2)) .* (4./(zB.^3) + 1./(zB.^2))) .* nu + ...
                (z.^2 - 1) .* zA ./ (nu * zB.^2);
            
            llg3 = [llg3, [diag(df11nu), diag(df12nu); diag(df12nu), diag(df22nu)]];
            
        end
  end
end


function [llg2, llg22] = lik_inputdependentT_fi(lik, y, f, param, z)
% LIK_INPUTDEPENDENTT_FI  Expected Fisher information matrix
% -E[ hessian of the log likelihood ] w.r.t. Y|f1, f2, sig2, nu
%
%  Description:        
%    LLG2 = LIK_INPUTDEPENDENTT_FI(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK. Returns the negative expectation of the hessian of the 
%    log likelihood with respect to PARAM. At the moment PARAM can be 
%    'latent' and 'latent+param'. FLLG2 returns the negative expectation
%    of second derivatives w.r.t. the latent  process f = (f1, f2) and
%    parameters of the of the likelihood function. 
%    In this case the matrix is sparse, therefore only the non-zero 
%    elements are returned in a way that is easy to build the sparse matrix
%    from them. This subfunction is needed when using Laplace approximation
%    or EP for inference with non-Gaussian likelihoods.
%
%  See also:
%    LIK_INPUTDEPENDENTT_LL, LIK_INPUTDEPENDENTT_LLG, LIK_INPUTDEPENDENTT_LLG3, GPLA_E

  n = size(y, 1);
  
  sig2 = lik.sig2;
  nu   = lik.nu;
  
  f  = f(:);
  % f1 = f(1:n); 
  f2 = f(n+1 : 2*n);
  
  sig = sig2 .* exp(f2);
  sig(sig < eps) = eps;
  sig(sig > 1e6) = 1e6;
  
  switch param
      case 'param'
          % ------ %
          
      case 'latent'
          % -E[d²l/df1²]
          d2f1 = 1./(sig.^2) .* (nu + 1)/(nu + 3);
          % d2f1 = 1./(sig.^2) .* (nu + 1)/(nu + 3);
          
          % -E[d²l/df1 df2] (on the log-scale)
          d2f1f2 = zeros(n, 1);
          
          % here we know that the expectation of the score function
          % is zero for well-behaved (integrable) r.v. E[|X|] < Inf.
          % -E[d²l/df2²] = -E[d²l/dsigmaf2² * sigmaf2²]
          d2f2 = 2*nu/(nu + 3) .* ones(n, 1);
          
          % return the non-zero elements of the expected Hessian (the matrix is
          % composed by diagonal blocks).
          llg2 = [d2f1, d2f1f2; d2f1f2, d2f2];
          llg2(isinf(llg2)) = sign(llg2(isinf(llg2))) .* realmax;
          
          llg22 = [diag(d2f1), diag(d2f1f2); diag(d2f1f2), diag(d2f2)];
          llg22(isinf(llg22)) = sign(llg22(isinf(llg22))) .* realmax;
          
      case 'latent+param'
          % ------ %
  end
end

function llg3 = lik_inputdependentT_dfi(lik, y, f, param, z)
% LIK_INPUTDEPENDENTT_LLG3  Derivatives of the matrix information matrix 
%
%  Description
%    DFI = LIK_INPUTDEPENDENTT_DFI(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, observations Y = y, censoring indicators Z = z and
%    latent values f and returns the derivatives of the expected fisher 
%    information matrix w.r.t. to PARAM. At the moment PARAM can be
%    only 'latent'. dfi is a vector with expected Fisher information 
%    derivatives. This  subfunction is needed when using Laplace
%    approximation based on the negative expected Hessian for inference
%    with non-Gaussian likelihoods.
%
%  See also
%    LIK_INPUTDEPENDENTT_LL, LIK_INPUTDEPENDENTT_LLG,
%    LIK_INPUTDEPENDENTT_LLG2, GPLA_E, GPLA_G

  n = size(y, 1);
  
  sig2 = lik.sig2;
  nu   = lik.nu;
  
  f  = f(:);
  % f1 = f(1:n); 
  f2 = f(n+1 : 2*n);
  
  sig = sig2 .* exp(f2);
  sig(sig < eps) = eps;
  sig(sig > 1e6) = 1e6;
  
  zrs = zeros(n, 1);
  
  switch param
    case 'param'
        % ----- %
        
    case 'latent'
        % - derivative of E[W] w.r.t. f1 - % 
        % d³l/df1 df1 df1
        df111 = zrs;
        
        % d³l/df1 df2 df1 
        df121 = zrs;
        
        % d³l/df2 df2 df1
        df221 = zrs;
            
        % - derivative of E[W] w.r.t. f2 - %
        % d³l/df1 df1 df2
        df112 = (-2./sig.^2) .* (nu + 1)/(nu + 3);
        
        % d³l/df1 df2 df2 
        df122 = zrs;
        
        % d³l/df2 df2 df2
        df222 = zrs;
        
        % rearranging the third derivatives for gpla_g.
        llg3 = zeros(2, 2, 2, n);
        
        % - derivative of E[W] w.r.t. f1 - %
        llg3(1, 1, 1, :) = df111;
        llg3(1, 2, 1, :) = df121;
        llg3(2, 1, 1, :) = llg3(1, 2, 1, :);
        llg3(2, 2, 1, :) = df221;
        
        
        % - derivative of E[W] w.r.t. f2 - %
        llg3(1, 1, 2, :) = df112;   
        llg3(1, 2, 2, :) = df122;
        llg3(2, 1, 2, :) = llg3(1, 2, 2, :);
        llg3(2, 2, 2, :) = df222;
        
    case 'latent2+param'
        
        llg3 = [];
        
        if ~isempty(lik.p.sig2)
            % - third derivatives of E[W] w.r.t. log(sig2) - %
            % d³l/df1 df1 dlog(sig2)
            df11s = (-2./sig.^2) .* (nu + 1)/(nu + 3);
            
            % d³l/df1 df2 dlog(sig2)
            df12s = zrs;
            
            % d³l/df2 df2 dlog(sig2)
            df22s = zrs;
            
            llg3 = [diag(df11s), diag(df12s); diag(df12s), diag(df22s)];
        end
            
        if ~isempty(lik.p.nu)
            % - third derivatives of E[W] w.r.t. log(nu) - %
            % d³l/df1 df1 dlog(nu)
            df11nu = 2./(sig.^2) .* nu ./ (nu + 3)^2;
            
            % d³l/df1 df2 dlog(nu)
            df12nu = zrs;
            
            % d³l/df2 df2 dlog(nu)
            df22nu = 6*nu/((nu + 3)^2) .* ones(n, 1);
            
            llg3 = [llg3, [diag(df11nu), diag(df12nu); diag(df12nu), diag(df22nu)]];
        end

  end
end

function [lpy, Ey, Vary] = lik_inputdependentT_predy(lik, Ef, Varf, yt, zt)
%LIK_INPUTDEPENDENTT_PREDY  Returns the predictive mean, variance and density of y
%
%  Description   
%    LPY = LIK_INPUTDEPENDENT_PREDY(LIK, EF, VARF YT, ZT)
%    Returns logarithm of the predictive density PY of YT, that is 
%        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the survival times YT, censoring indicators ZT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_INPUTDEPENDENT_PREDY(LIK, EF, VARF) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction
%    is needed when computing posterior predictive distributions for 
%    future observations.
%        
%
%  See also
%    GPLA_PRED, GPEP_PRED, GPMC_PRED

sig2 = lik.sig2;
nu   = lik.nu;

% expectations;
Ef = Ef(:);
ntest = size(Ef, 1)/2;

Ef1 = Ef(1:ntest); 
Ef2 = Ef(ntest+1 : end); 

% variances
if size(Varf,2) == size(Varf,1)
    Varf1 = diag(Varf(1:ntest, 1:ntest));
    Varf2 = diag(Varf(ntest + 1:end, ntest + 1:end));
    
% elseif size(Varf, 2) == 1
%     Varf = diag(Varf);
%     Varf1 = Varf(1:ntest); 
%     Varf2 = Varf(ntest+1 : end);
    
else
    Varf = diag(Varf);
    Varf1 = Varf(1:ntest); 
    Varf2 = Varf(ntest+1 : end);

end

% covariances
% ind = (((ntest+1):2*ntest) - 1) * 2*ntest + (1:ntest);
% Covf1f2 = Varf(ind);

Ey = []; Vary = [];

if nargout > 1
    if nu > 1
        Ey = Ef1;
        if nu > 2
            Vary = Varf1 + sig2 * exp(2*Ef2 + 2*Varf2) .* nu./(nu - 2);
        else
            warning('Variance of Student''s-t distribution does not exist for nu <= 2');
            nu = 2.00001;
            Vary = Varf1 + sig2 * exp(2*Ef2 + 2*Varf2) .* nu./(nu - 2);
            %Vary = NaN + Varf;
        end
    else
        Ey = NaN + Ef;
        warning('Y|f1, f2, nu is not integrable. No first and second order moments exist');
    end
end

% Evaluate the posterior predictive densities in the observed points
lpy = zeros(length(yt), 1);
if ~isempty(yt)
    if (size(Ef, 1) == size(Varf, 2)) %&& (size(Ef, 1)/2 == size(yt, 1)) && size(yt, 2) == 1
        for i2 = 1:ntest
            m1 = Ef1(i2);
            m2 = Ef2(i2); 
            
            % Variances, covariance and inverse Σ
            Varf1 = Varf(i2, i2);
            Varf2 = Varf(ntest + i2, ntest + i2);
            Covf1f2 = Varf(i2, ntest + i2);
            
            detS = Varf1*Varf2 - Covf1f2^2;
            invS = [Varf2 -Covf1f2; -Covf1f2  Varf1]/detS;
            
            % standart deviations for the limits of the integration
            s1 = sqrt(Varf1);
            s2 = sqrt(Varf2);
            
            % Function handle for st(y|nu, f1, sig*exp(f2)) * Gaussian(f1, f2| Ef12, Cov12)
            pd = @(f1, f2) t_pdf(yt(i2), nu, f1, sig2*exp(f2)) ...
                .* 1/(2*pi * sqrt(detS)) .* exp(-0.5 .*((f1 - m1).^2 .* invS(1, 1) + ...
                2.*(f1 - m1).*(f2 - m2).*invS(1, 2) + invS(2, 2).*(f2 - m2).^2));
            
            % integrate w.r.t. latent values
            lpy(i2) = log(dblquad(pd, m1 - 6.*s1, m1 + 6.*s1, m2 - 6.*s2, m2 + 6.*s2));
        end
    end
end

end

function prctys = lik_inputdependentT_predprcty(lik, Ef, Varf, zt, prcty)
%LIK_inputdependentT_PREDPRCTY  Returns the percentiles of predictive density of y
%
%  Description         
%    PRCTY = LIK_inputdependentT_PREDPRCTY(LIK, EF, VARF YT, ZT)
%    Returns percentiles of the predictive density PY of YT. This
%    subfunction is needed when using function gp_predprcty.
%
%  See also 
%    GP_PREDPCTY

  sig2 = lik.sig2;
  nu = lik.nu;
  
  % percentiles
  nt = size(Ef, 1);
  prcty = prcty/100;
  
  % expectations;
  Ef = Ef(:);
  ntest = size(Ef, 1)/2;
  
  Ef1 = Ef(1:ntest); 
  Ef2 = Ef(ntest+1 : end); 
  
  % variances
  if size(Varf,2) == size(Varf,1)
      Varf1 = diag(Varf(1:ntest, 1:ntest));
      Varf2 = diag(Varf(ntest + 1:end, ntest + 1:end));
      
  %  elseif size(Varf, 2) == 1
  %  Varf1 = Varf(1:ntest); 
  %  Varf2 = Varf(ntest+1 : end); 
      
  else
      Varf = diag(Varf);
      Varf1 = Varf(1:ntest); 
      Varf2 = Varf(ntest+1 : end);
      
  end
  
  % covariances
  ind = (((ntest+1):2*ntest) - 1) * 2*ntest + (1:ntest);
  Covf1f2 = Varf(ind);
  
  % Vary = nu./(nu - 2) .* sig2 + Varf;
  % we may not use the total variance property here ...  
  if nu <= 2
      warning('Variance of Student''s-t distribution does not exist for nu <= 2');
  end

  % compute required percentile 
  prctys = zeros(ntest, numel(prcty));
  for i1 = 1:numel(prcty)
      for i2 = 1:ntest
          % expectations 
          m1 = Ef1(i2); 
          m2 = Ef2(i2);    
          
          % variance, covariance
          Vf1 = Varf1(i2); 
          Vf2 = Varf2(i2);
          Cvf1f2 = Covf1f2(i2);
          
          detS = Vf1*Vf2 - Cvf1f2^2;
          invS = [Vf2 -Cvf1f2; -Cvf1f2  Vf1]/detS;
          
          % lower and upper limits of integration
          l1 = m1 - 6*sqrt(Vf1);   u1 = m1 + 6*sqrt(Vf1); 
          l2 = m2 - 6*sqrt(Vf2);   u2 = m2 + 6*sqrt(Vf2);
          
          % Function handle for student-T(x|f1, σ2*exp(f2), ν) * Gaussian(f1, f2| μ, Σ)
          pd = @(x, f1, f2)  tcdf((x - f1)./(sig2*exp(f2)), nu) ...
              .* 1/(2*pi * sqrt(detS)) .* exp(-0.5 .*((f1 - m1).^2 .* invS(1, 1) + ...
              2.*(f1 - m1).*(f2 - m2).*invS(1, 2) + invS(2, 2).*(f2 - m2).^2));
          
          % f(x) = p - P(X <= x)
          pr = @(x) prcty(i1) - dblquad(@(f1, f2) pd(x, f1, f2),  l1, u1, l2, u2);
          
          % find x such that f(x) = 0;
          prctys(i2, i1) = fzero(pr, m1);      
          % prctys(i2, i1) = fzero(pr, [m1-7*sqrt(Varf1)  m1+7*sqrt(Varf1)]);      
      end
  end
end


function reclik = lik_inputdependentT_recappend(reclik, ri, lik)
%RECAPPEND  Record append
%  Description
%    RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF) takes old
%    covariance function record RECCF, record index RI, RECAPPEND
%    returns a structure RECCF. This subfunction is needed when 
%    using MCMC sampling (gp_mc).

  if nargin == 2
    % Initialize the record
    reclik.type = 'inputdependentt';
    reclik.nondiagW = true;

    % Initialize parameters
    reclik.nu = [];
    reclik.sig2 = [];

    % Set the function handles
    reclik.fh.pak = @lik_inputdependentT_pak;
    reclik.fh.unpak = @lik_inputdependentT_unpak;
    reclik.fh.lp = @lik_inputdependentT_lp;
    reclik.fh.lpg = @lik_inputdependentT_lpg;
    reclik.fh.ll = @lik_inputdependentT_ll;
    reclik.fh.llg = @lik_inputdependentT_llg;    
    reclik.fh.llg2 = @lik_inputdependentT_llg2;
    reclik.fh.llg3 = @lik_inputdependentT_llg3;
    reclik.fh.predy = @lik_inputdependentT_predy;
    reclik.fh.predprcty = @lik_inputdependentT_predprcty;
    reclik.fh.recappend = @lik_inputdependentT_recappend;
    
    reclik.p.nu = [];
    if ~isempty(ri.p.nu)
        reclik.p.nu = ri.p.nu;
    end
    
    reclik.p.sig2 = [];
    if ~isempty(ri.p.sig2)
        reclik.p.sigma2 = ri.p.sig2;
    end
    
  else
      % Append to the record
      likp = lik.p;
      
      % record sig2
      reclik.sig2(ri, :) = lik.sig2;
      if isfield(likp, 'sig2') && ~isempty(likp.sig2)
          reclik.p.sig2 = likp.sig2.fh.recappend(reclik.p.sig2, ri, likp.sig2);
      end
      
      % record nu
      reclik.nu(ri, :) = lik.nu;
      if isfield(likp,'nu') && ~isempty(likp.nu)
          reclik.p.nu = likp.nu.fh.recappend(reclik.p.nu, ri, likp.nu);
      end
  end

end
