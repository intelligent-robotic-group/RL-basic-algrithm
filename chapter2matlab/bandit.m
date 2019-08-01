classdef bandit < handle
    %BANDIT 10_arms bandit
    %   Author: zhongjie
    %   Date: 2019.5.8 
    properties
        k
        step_size
        sample_averages
        indices
        time
        average_reward 
        true_reward
        epsilon
        initial
        alfa
        q_true
        q_estimation
        action_count
        best_action
    end
    
    methods
        function obj = bandit(varargin)
            %BANDIT 构造此类的实例
            obj.k=10;
            obj.step_size=0.1;
            obj.sample_averages=true;
            obj.indices=1:1:obj.k;
            obj.time=0;
            obj.average_reward=0;
            obj.true_reward=0;
            obj.epsilon=0.1;
            obj.initial=0;
            obj.alfa=0.1;
            if length(varargin)>0
                for i=1:1:length(varargin)
                    switch inputname(i)
                        case 'k'
                            obj.k=varargin{i};
                            obj.indices=1:1:obj.k;
                        case 'sample_averages'
                            obj.sample_averages=varargin{i};
                        case 'epsilon'
                            obj.epsilon=varargin{i};
                    end
                end
            end    
        end
        
        function reset(obj)
%real reward for each action
            obj.q_true=obj.true_reward+randn(1,obj.k);
%  estimation for each action
            obj.q_estimation = zeros(1,obj.k) + obj.initial;
%   # of chosen times for each action
            obj.action_count = zeros(1,obj.k);
            
            obj.best_action = find(obj.q_true==max(obj.q_true));          
        end
        
        function action=act(obj)
            if rand()<obj.epsilon
                action = unidrnd(obj.k);
            else
                action = find(obj.q_estimation==max(obj.q_estimation));
                if length(action)>1
                    action=action(1);
                end
            end
        end
        
        function reward=step(obj,action)
            obj.q_true=normrnd(0,0.01,1,obj.k) + obj.q_true;
            obj.best_action=find(obj.q_true==max(obj.q_true));
            if obj.best_action>1
                obj.best_action=obj.best_action(1);
            end
            reward = randn() + obj.q_true(action);
            obj.time=obj.time+1;
            obj.action_count(action)=obj.action_count(action)+1;
            if obj.sample_averages
                obj.q_estimation(action)=obj.q_estimation(action)+1.0 / obj.action_count(action)* (reward - obj.q_estimation(action));
            else
                obj.q_estimation(action) =obj.q_estimation(action)+obj.alfa*(reward-obj.q_estimation(action));
            end
        end  
    end
end

