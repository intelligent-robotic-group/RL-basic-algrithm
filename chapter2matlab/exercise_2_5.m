
clear all;
clc;
bandits=bandit();
runs=2000;time=10000;
best_action_counts=zeros(runs,time);
rewards = zeros(runs,time);
for i=1:1:runs
    i
    bandits.reset();
    for t=1:1:time
        action = bandits.act();
        reward = bandits.step(action);
        rewards(i, t) = reward;
        if action==bandits.best_action
            best_action_counts(i, t) = 1;
        end
    end
end
a=mean(rewards);
b=mean(best_action_counts);
figure(1);
subplot 211;
plot(a);hold on;
subplot 212;
plot(b);hold on;


sample_averages=false;
bandits=bandit(sample_averages);
runs=2000;time=10000;
best_action_counts=zeros(runs,time);
rewards = zeros(runs,time);
for i=1:1:runs
    i
    bandits.reset();
    for t=1:1:time
        action = bandits.act();
        reward = bandits.step(action);
        rewards(i, t) = reward;
        if action==bandits.best_action
            best_action_counts(i, t) = 1;
        end
    end
end
c=mean(rewards);
d=mean(best_action_counts);
figure(1);
subplot 211;
plot(c);
subplot 212;
plot(d);
            
        
        
        
        
        
        
        
        
        
        

