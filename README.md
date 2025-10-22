# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.

## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.

## POLICY ITERATION ALGORITHM
Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.


## POLICY IMPROVEMENT FUNCTION
#### Name:- MUKESH P
#### Register Number:- 212222240068
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = np.argmax(Q, axis=1)

    return new_pi

def callable_policy(pi_array):
    return lambda s: pi_array[s]

pi_2_array = policy_improvement(V1, P)
pi_2 = callable_policy(pi_2_array)

print("Name:  MUKESH P   ")
print("Register Number:212222240068       ")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)


```
## POLICY ITERATION FUNCTION
#### Name:- MUKESH P
#### Register Number:- 212222240068
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    # Write your code here for policy iteration
    n_states = len(P)
    n_actions = len(P[0])

    # Initialize a random policy
    pi = np.random.randint(0, n_actions, n_states)

    while True:
        # Policy Evaluation
        V = np.zeros(n_states, dtype=np.float64)
        while True:
            prev_V = V.copy()
            for s in range(n_states):
                v = 0
                action = pi[s]
                for prob, next_state, reward, done in P[s][action]:
                    v += prob * (reward + gamma * prev_V[next_state] * (not done))
                V[s] = v
            if np.max(np.abs(prev_V - V)) < theta:
                break

        # Policy Improvement
        new_pi = np.zeros(n_states, dtype=np.int64)
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            for a in range(n_actions):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            new_pi[s] = np.argmax(Q[s])

        # Check for policy convergence
        if np.array_equal(pi, new_pi):
            break
        pi = new_pi

    return V, callable_policy(pi)
optimal_V, optimal_pi = policy_iteration(P)


print("Name:   MUKESH P  ")
print("Register Number:  212222240068       ")
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)






```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy

#### Policy
<img width="590" height="167" alt="31" src="https://github.com/user-attachments/assets/d2eb01d5-ff5e-4093-b585-a2f655d3f4cf" />


#### Value function
<img width="545" height="162" alt="32" src="https://github.com/user-attachments/assets/d51f4d81-0efd-4d02-b5f5-046adee3ba9d" />


#### success rate
<img width="817" height="25" alt="image" src="https://github.com/user-attachments/assets/1dec0fb4-fd40-4258-aed3-e930e8d22a72" />

</br>

### 2. Policy, Value function and success rate for the Improved Policy

#### Policy
<img width="627" height="160" alt="34" src="https://github.com/user-attachments/assets/37ca2334-1069-40b7-96b3-fb832a3e5260" />

#### Value function
<img width="621" height="156" alt="35" src="https://github.com/user-attachments/assets/0873c5ed-8998-45a7-9fbd-20afe8bb7076" />


#### success rate
<img width="817" height="38" alt="image" src="https://github.com/user-attachments/assets/9c2690cc-7497-4420-96dc-15c5a6383d05" />
</br>

### 3. Policy, Value function and success rate after policy iteration

</br>

#### Policy
<img width="562" height="173" alt="image" src="https://github.com/user-attachments/assets/995171c0-db40-47f4-a8e8-e486a0d3f6bf" />

#### success rate
<img width="831" height="25" alt="image" src="https://github.com/user-attachments/assets/0284cd8b-929c-43a7-853f-4be0fa07a1a3" />

#### Value function
<img width="908" height="130" alt="39" src="https://github.com/user-attachments/assets/5e3091b8-3065-4efb-b81c-1e2e17e8ed24" />





## RESULT:

Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
