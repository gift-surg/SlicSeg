% Author: Stefan Stavrev 2013

% Description: Fitting mixture of Gaussians.
% Input: x       - each row is one datapoint.
%        K       - number of Gaussians in the mixture.
%        precision - the algorithm stops when the difference between
%                    the previous and the new likelihood is < precision.
%                    Typically this is a small number like 0.01.
% Output:
%        lambda  - lambda(k) is the weight for the k-th Gaussian.
%        mu      - mu(k,:) is the mean for the k-th Gaussian.
%        sig     - sig{k} is the covariance matrix for the k-th Gaussian.
function [lambda, mu, sig] = fit_mog (x, K, precision)
    % Initialize all values in lambda to 1/K.
    lambda = repmat (1/K, K, 1);

    % Initialize the values in mu to K randomly chosen unique datapoints.
    I = size (x, 1);
    K_random_unique_integers = randperm(I);
    K_random_unique_integers = K_random_unique_integers(1:K);
    mu = x (K_random_unique_integers,:);

    % Initialize the variances in sig to the variance of the dataset.
    sig = cell (1, K);
    dimensionality = size (x, 2);
    dataset_mean = sum(x,1) ./ I;
    dataset_variance = zeros (dimensionality, dimensionality);
    for i = 1 : I
        mat = x (i,:) - dataset_mean;
        mat = mat' * mat;
        dataset_variance = dataset_variance + mat;
    end
    dataset_variance = dataset_variance ./ I;
    for i = 1 : K
        sig{i} = dataset_variance;
    end
    
    % The main loop.
    iterations = 0;    
    previous_L = 1000000; % just a random initialization
    while true
        % Expectation step.
        l = zeros (I,K);
        r = zeros (I,K);
        % Compute the numerator of Bayes' rule.
        for k = 1 : K
            l(:,k) = lambda(k) * mvnpdf (x, mu(k,:), sig{k});
        end
        
        % Compute the responsibilities by normalizing.
        s = sum(l,2);        
        for i = 1 : I
            r(i,:) = l(i,:) ./ s(i);
        end

        % Maximization step.
        r_summed_rows = sum (r,1);
        r_summed_all = sum(sum(r,1),2);
        for k = 1 : K
            % Update lambda.
            lambda(k) = r_summed_rows(k) / r_summed_all;

            % Update mu.
            new_mu = zeros (1,dimensionality);
            for i = 1 : I
                new_mu = new_mu + r(i,k)*x(i,:);
            end
            mu(k,:) = new_mu ./ r_summed_rows(k);

            % Update sigma.
            new_sigma = zeros (dimensionality,dimensionality);
            for i = 1 : I
                mat = x(i,:) - mu(k,:);
                mat = r(i,k) * (mat' * mat);
                new_sigma = new_sigma + mat;
            end
            sig{k} = new_sigma ./ r_summed_rows(k);
        end
        
        % Compute the log likelihood L.
        temp = zeros (I,K);
        for k = 1 : K
            temp(:,k) = lambda(k) * mvnpdf (x, mu(k,:), sig{k});
        end
        temp = sum(temp,2);
        temp = log(temp);        
        L = sum(temp);  
        %disp(L);
 
        iterations = iterations + 1;        
        %disp([num2str(iterations) ': ' num2str(L)]);
        if abs(L - previous_L) < precision
            %msg = [num2str(iterations) ' iterations, log-likelihood = ', ...
                %num2str(L)];
            %disp(msg);
            break;
        end
        
        previous_L = L;
    end
end