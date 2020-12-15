'''
Sites and pages that are helpful:
DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
DDPG Code: https://github.com/germain-hug/Deep-RL-Keras/tree/master/DDPG
Keras.Gradient: https://www.tensorflow.org/api_docs/python/tf/gradients
Keras.Function: https://www.tensorflow.org/api_docs/python/tf/keras/backend/function
Zip: https://www.geeksforgeeks.org/zip-in-python/
TensorFlow version used: less than 2.0.0 as then tf.gradients dosent work
'''

# Classes
from ADL_Actor import ADL_Actor
from ADL_Critic import ADL_Critic
from D_S_Actor import D_S_Actor
from D_S_Critic import D_S_Critic
from A_R_Actor import A_R_Actor
from P_Q_Actor import P_Q_Actor
from A_R_P_Q_Critic import A_R_P_Q_Critic
from ReplayBuffer import ReplayBuffer
from ReplayBuffer_A_R_P_Q import ReplayBuffer_A_R_P_Q

# Libraries
import tensorflow as tf
import collections
import random
import numpy as np
import math
import copy

'''
    To Dos
    Adding Gamma to all rewards for the Critics
    Add arguments to all the classes in the dictionary
    Writing the get_renewable and update_adl function and get_demand function
    Create the Q(s1,s2,a1,a2,a3,a4) - a1'[Loss Values] for the A_R_Network 
    Incorporate ADL
    Use the inverse tanh function for the linear transformation.  DONE. GET IT CHECKED.
    Modify Replay Buffer Class so that you dont have to store the next state seperately. DONE
'''

'''
    For the implementation of a linear curve fit for the tanh function, we assume the highest values are:
    np.arctanh(0.9999999999999999)= 18.714973875118524 & np.arctanh(-0.9999999999999999) = -18.714973875118524
    And the assumption is those many digits are only going to be in the output (No idea if this is true)
'''

def get_renewable(index, time):
    lamb = [[2.667e-07, 0.541, 6.5965, 4.3712], [8.8281, 10.2997, 9.8301, 9.7514], [8.8281, 10.2997, 9.8301, 9.7514]]
    energy = np.random.poisson(lam=lamb[index][time], size=1)
    energy = min([8, energy])  # clipping the value so that it can't exceede 8
    energy = int(math.floor(energy))
    return energy


# Write this function
def get_demand(index, time):
    non_adl = [3, 4, 5, 6]
    prob_non_adl = [[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2], [0.1, 0.3, 0.4, 0.2], [0.2, 0.3, 0.1, 0.4]]
    demand = np.random.choice(non_adl, p=prob_non_adl[time - 1])
    return int(demand)


# Write this function
def update_adl(index, time):
    return 0


class Main:
    '''
        States and Actions of the main controller:
            ADL:
                States: [ time, battery, renewable-energy, ADL_State, current-demand, grid-price]
                ACtions: ADL-Demand to be fulfilled in that time step
            D_S:
                States: [ time, battery, renewable-energy, ADL_State, current-demand, grid-price, ADL_action]
                Actions: Energy demand or energy supply that has to performed at that time step

        States and Actions of the Sub-Controllers (Roughly):
            A_R:
                States: [ internal time, n-sized vector that indicates whose a buyer or seller, p and q for all the other agents]
                Actions: Accept or Reject Vector
            P_Q:
                States:  [ internal time, n-sized vector that indicates whose a buyer or seller, Accept_reject vector, p and q for all the other agents, ]
                Actions: Price and Quantity Vector
    '''

    no_of_agents = 3
    k = 5
    grid_price = 20
    lower_price = grid_price - 5
    names = []  # Names for the agents
    agent = []
    total_no_of_iterations = 1000000
    time_steps_per_day = 4
    no_of_sub_transactions = 0
    c1 = 20
    # Note the arguments for the objects have to be entered
    for j in range(total_no_of_iterations):
        if j == 0:
            for i in range(no_of_agents):
                agent.append({'index': i, 'name': names[i], 'ADL_Actor': ADL_Actor(), 'ADL_Critic': ADL_Critic(),
                              'D_S_Actor': D_S_Actor(5, 0.01, 0.1), 'D_S_Critic': D_S_Critic(5, 1),
                              'A_R_Actor': A_R_Actor((3 + 3*no_of_agents),0.01, 0.1, no_of_agents), 'P_Q_Actor': P_Q_Actor((3 + 4*no_of_agents),0.01, 0.1, no_of_agents), 'A_R_P_Q_Critic': A_R_P_Q_Critic((3 + 3*no_of_agents), (3 + 4*no_of_agents), no_of_agents, no_of_agents, no_of_agents),
                              'Battery': 5, 'Renewable': 0,
                              'Demand': 0, 'ADL_Value': 0, 'ADL_State': [], 'ADL_Action': [], 'D_S_State': [],
                              'D_S_Action': [], 'Current_Demand': 0, 'Main_Reward': 0,
                              'A_R_State': [], 'A_R_Action': [], 'P_Q_State': [], 'P_Q_Action': [], 'Via_Matrix': 0,
                              'Sub_Reward': 0,
                              'A_R_Vector': np.zeros(no_of_agents), 'P_Vector': np.zeros(no_of_agents),
                              'Q_Vector': np.zeros(no_of_agents), 'Status': [], 'Customised_Status': [],
                              'D_S_Buffer': ReplayBuffer(), 'A_R_P_Q_Buffer': ReplayBuffer_A_R_P_Q()})

        '''
            Ignore ADL for now
            Hence the Demand state is: [time, battery, renewable-energy, current-demand, grid-price]  
        '''

        '''
                Difference between demand and current-demand is that current demand is updated over time, wheras demand stays constant
                Difference between Status and Customised-Status is that status is customized particularly for a buyer or a seller 
                Via_Matrix stands for Viability (Whether the transactions are valid or not) 
        '''

        buyers = []
        sellers = []
        nothing = []

        # Getting the D_S_State and storing it in respective variables and replay buffers
        for i in range(no_of_agents):
            status = []  # Whether the agents are buyers or sellers or nothing (1: Sellers, 0:Buyers, -1: Nothing)

            agent[i]['D_S_State'] = []
            agent[i]['D_S_Action'] = 0
            agent[i]['Main_Reward'] = 0
            agent[i]['Sub_Reward'] = 0
            agent[i]['Renewable'] = get_renewable(i, j % time_steps_per_day)
            agent[i]['ADL_Value'] = update_adl(i, j % time_steps_per_day)
            agent[i]['Demand'] = get_demand(i, j % time_steps_per_day)
            agent[i]['D_S_State'] = [j % time_steps_per_day, agent[i]['Battery'], agent[i]['Renewable'],
                                     agent[i]['Demand'], grid_price]
            agent[i]['D_S_Buffer'].store_transition_state(agent[i]['D_S_State'])

        # Getting the D_S_Actions and storing it in respective variables and replay buffers using the constraints
        for i in range(no_of_agents):
            agent[i]['D_S_Action'] = (np.arctanh(
                agent[i]['D_S_Actor'].action(agent[i]['D_S_State'])) + 18.714973875118524) * \
                                     (agent[i]['Battery'] + agent[i]['Demand']) / 37.42994775023705 + agent[i][
                                         'Renewable'] - \
                                     agent[i]['Demand']
            agent[i]['D_S_Buffer'].store_transition_action(agent[i]['D_S_Actor'].action(agent[i]['D_S_State']))
            agent[i]['Battery'] = max(0,
                                      agent[i]['Renewable'] + agent[i]['Battery'] - agent[i]['D_S_Action'] - agent[i][
                                          'Demand'])
            if agent[i]['D_S_Action'] < 0:
                status.append(0)
            if agent[i]['D_S_Action'] == 0:
                status.append(-1)
            else:
                status.append(1)

        for i in range(no_of_agents):
            agent[i]['Status'] = status
            status_1 = status
            agent[i]['Current_Demand'] = agent[i]['D_S_Action']
            if agent[i]['D_S_Action'] < 0:
                for l in range(len(status)):
                    if status_1[i] == -1:
                        status_1[i] = 0
            elif agent[i]['D_S_Action'] == 0:
                pass
            else:
                for l in range(len(status)):
                    if status_1[i] == -1:
                        status_1[i] = 1
            agent[i]['Customised_Status'] = status_1

        # Separating into Buyers, Sellers or Nothing and updating the Via_Matrix
        for i in range(no_of_agents):
            if agent[i]['D_S_Action'] < 0:
                agent[i]['Via_Matrix'] = np.diag(agent[i]['Customised_Status'])
                buyers.append(agent[i])
            if agent[i]['D_S_Action'] == 0:
                agent[i]['Via_Matrix'] = np.diag(agent[i]['Customised_Status'])
                nothing.append(agent[i])
            else:
                agent[i]['Via_Matrix'] = np.ones(len(agent[i]['Customised_Status'])) - np.diag(
                    agent[i]['Customised_Status'])
                agent[i]['Customised_Status'] = 1 - agent[i]['Customised_Status']  # changed
                sellers.append(agent[i])

        '''
        States and Actions of the Sub-Controllers (Roughly):
            A_R:
                States: [ internal time, demand, current-demand, status-vector, p-vector for all the other agents, q-vector for all the other agents]
                Actions: Accept or Reject Vector
            P_Q:
                States:  [ internal time, demand, current-demand, status-vector, p-vector for all the other agents, q-vector for all the other agents, A_R_Vector]
                Actions: Price and Quantity Vector
        '''

        # Sub_Transactions
        for k in range(no_of_sub_transactions):

            for m in range(len(sellers)):
                sellers[m]['A_R_P_Q_Buffer'].store_transition_via_matrix(sellers[m]['Via_Matrix'])
                sellers[m]['A_R_P_Q_Buffer'].store_transition_customized_status(sellers[m]['Customized_Status'])

            for m in range(len(buyers)):
                buyers[m]['A_R_P_Q_Buffer'].store_transition_via_matrix(buyers[m]['Via_Matrix'])
                buyers[m]['A_R_P_Q_Buffer'].store_transition_customized_status(buyers[m]['Customized_Status'])

            # A_R for Sellers
            # If the Sub-Transaction time is 0, Ignore the A_R_Network, the action would be a zero everywhere
            if k == 0:
                for m in range(len(sellers)):
                    sellers[m]['P_Q_State'] = [k, sellers[m]['Demand'], sellers[m]['Current_Demand']]
                    sellers[m]['P_Q_State'].extend(sellers[m]['Customised_Status'])
                    sellers[m]['P_Q_State'].extend(sellers[m]['P_Vector'])
                    sellers[m]['P_Q_State'].extend(sellers[m]['Q_Vector'])
                    sellers[m]['A_R_State'] = sellers[m]['P_Q_State']

                    sellers[m]['A_R_P_Q_Buffer'].store_transition_state_a_r(sellers[m]['P_Q_State'])
                    sellers[m]['A_R_P_Q_Buffer'].store_transition_action_a_r(np.zeros(no_of_agents))

                    sellers[m]['P_Q_State'].extend(np.zeros(no_of_agents))
                    sellers[m]['A_R_P_Q_Buffer'].store_transition_state_p_q(sellers[m]['P_Q_State'])

            else:
                for m in range(len(sellers)):
                    sellers[m]['A_R_State'] = [k, sellers[m]['Demand'], sellers[m]['Current_Demand']]
                    sellers[m]['A_R_State'].extend(sellers[m]['Customised_Status'])
                    sellers[m]['A_R_State'].extend(sellers[m]['P_Vector'])
                    sellers[m]['A_R_State'].extend(sellers[m]['Q_Vector'])

                    sellers[m]['A_R_P_Q_Buffer'].store_transition_state_a_r(sellers[m]['A_R_State'])

                    sellers[m]['A_R_Action'] = np.matmul(sellers[m]['Via_Matrix'],
                                                         sellers[m]['A_R_Actor'].action(sellers[m]['A_R_State']))
                    sellers[m]['A_R_Vector'] = np.asarray(sellers[m]['A_R_Action']) > 0.5

                    sellers[m]['P_Q_State'] = sellers[m]['A_R_State']
                    sellers[m]['P_Q_State'].extend(sellers[m]['A_R_Action'])

                    sellers[m]['A_R_P_Q_Buffer'].store_transition_action_a_r(sellers[m]['A_R_Action'])
                    sellers[m]['A_R_P_Q_Buffer'].store_transition_state_p_q(sellers[m]['P_Q_State'])

                    total_q = 0
                    reward = 0

                    for n in range(len(buyers)):
                        # The P and Q vector of the Sellers have to be the P and Q values of each buyer. That means that
                        # the  each P would be positive and each Q would be negative
                        buyers[n]['Sub_Reward'] = buyers[n]['Sub_Reward'] + sellers[m]['Q_Vector'][buyers[n]['index']] * \
                                                  sellers[m]['P_Vector'][buyers[n]['index']] * \
                                                  sellers[m]['A_R_Vector'][buyers[n]['index']]

                        # The negative sign makes sense as the values of the Q_vector are negative and the current demand
                        # is positive
                        buyers[n]['Current_Demand'] = min(0, buyers[n]['Current_Demand'] - sellers[m]['Q_Vector'][
                            buyers[n]['index']] * sellers[m]['A_R_Vector'][buyers[n]['index']])

                        # The negative sign makes sense as the values of the Q_vector are negative
                        total_q = total_q - sellers[m]['Q_Vector'][buyers[m]['index']] * sellers[m]['A_R_Vector'][
                            buyers[n]['index']]

                        # The negative sign makes sense as the values of the Q_vector are negative
                        reward = reward - sellers[m]['Q_Vector'][buyers[n]['index']] * sellers[m]['P_Vector'][
                            buyers[n]['index']] * sellers[m]['A_R_Vector'][buyers[n]['index']]

                    if total_q > sellers[m]['Current_Demand']:
                        # The reward is decreased as the excess energy sold is bought from the main grid.
                        reward = reward - (total_q - sellers[m]['Current_Demand']) * grid_price
                        sellers[m]['Sub_Reward'] = reward
                        sellers[m]['Current_Demand'] = 0

                    else:
                        sellers[m]['Sub_Reward'] = reward
                        sellers[m]['Current_Demand'] = sellers[m]['Current_Demand'] - total_q

            # P_Q_Action for Sellers

            for m in range((len(sellers))):
                action_p, action_q, action_q_traded = sellers[m]['P_Q_actor'].action(sellers[m]['P_Q_State'])

                action_p = np.matmul(sellers[m]['Via_Matrix'], action_p)
                action_q = np.matmul(sellers[m]['Via_Matrix'], action_q) / np.matmul(
                    np.matmul(sellers[m]['Via_Matrix'], action_q), sellers[m]['Customised_Status'])
                action_q_traded = action_q_traded

                sellers[m]['A_R_P_Q_Buffer'].store_transition_action_p_q(action_p, action_q, action_q_traded)
                action_q = ((np.arctanh(action_q_traded) + 18.714973875118524) / 37.42994775023705) * action_q * \
                           sellers[m]['Current_Demand']

                for n in range(len(buyers)):
                    buyers[n]['P_Vector'][sellers[m]['index']] = action_p[buyers[n]['index']]
                    buyers[n]['Q_Vector'][sellers[m]['index']] = action_q[buyers[n]['index']]

            # A_R_Action for Buyers
            for m in range(len(buyers)):
                buyers[m]['A_R_State'] = [k, buyers[m]['Demand'], buyers[m]['Current_Demand']]
                buyers[m]['A_R_State'].extend(buyers[m]['Customised_Status'])
                buyers[m]['A_R_State'].extend(buyers[m]['P_Vector'])
                buyers[m]['A_R_State'].extend(buyers[m]['Q_Vector'])

                buyers[m]['A_R_P_Q_Buffer'].store_transition_state_a_r(buyers[m]['A_R_State'])

                buyers[m]['A_R_Action'] = np.matmul(buyers[m]['Via_Matrix'],
                                                    buyers[m]['A_R_Actor'].action(buyers[m]['A_R_State']))
                buyers[m]['A_R_Vector'] = np.asarray(buyers[m]['A_R_Action']) > 0.5

                buyers[m]['P_Q_State'] = buyers[m]['A_R_State']
                buyers[m]['P_Q_State'].extend(buyers[m]['A_R_Action'])

                buyers[m]['A_R_P_Q_Buffer'].store_transition_action_a_r(buyers[m]['A_R_Action'])
                buyers[m]['A_R_P_Q_Buffer'].store_transition_state_p_q(buyers[m]['P_Q_State'])

                total_q = 0
                reward = 0

                for n in range(len(sellers)):
                    # The P and Q vector of the buyers have to be the P and Q values of each seller. That means that
                    # the  each P would be positive and each Q would be positive
                    sellers[n]['Sub_Reward'] = sellers[n]['Sub_Reward'] + buyers[m]['Q_Vector'][sellers[n]['index']] * \
                                               buyers[m]['P_Vector'][sellers[n]['index']] * \
                                               buyers[m]['A_R_Vector'][sellers[n]['index']]

                    # The negative sign makes sense as the values of the Q_vector are positive and the current demand
                    # is positive
                    sellers[n]['Current_Demand'] = max(0, sellers[n]['Current_Demand'] - buyers[m]['Q_Vector'][
                        sellers[n]['index']] * buyers[m]['A_R_Vector'][sellers[n]['index']])

                    # The positive sign makes sense as the values of the Q_vector are positive
                    total_q = total_q + buyers[m]['Q_Vector'][sellers[n]['index']] * buyers[m]['A_R_Vector'][
                        sellers[n]['index']]

                    # The negative sign makes sense as the values of the Q_vector are positive
                    reward = reward - buyers[m]['Q_Vector'][sellers[n]['index']] * buyers[m]['P_Vector'][
                        sellers[n]['index']] * \
                             buyers[m]['A_R_Vector'][sellers[n]['index']]

                if (-total_q) < buyers[m]['Current_Demand']:
                    # buyers[m]['Current_Demand'] is negative and total_q is positive. Reward is increased as the excess
                    # energy is sold back to the main grid.
                    reward = reward + (total_q + buyers[m]['Current_Demand']) * lower_price
                    buyers[m]['Sub_Reward'] = reward
                    buyers[m]['Current_Demand'] = 0

                else:
                    buyers[m]['Sub_Reward'] = reward

                    buyers[m]['Current_Demand'] = buyers[m]['Current_Demand'] + total_q

            # P_Q_Action for buyers If its the last sub time step, all the p actions and q_actions are 0 and there is
            # no need of updating the sellers

            for m in range(len(buyers)):

                if k == (no_of_sub_transactions - 1):
                    buyers[m]['A_R_P_Q_Buffer'].store_transition_action_p_q(np.zeros(no_of_agents),
                                                                            np.zeros(no_of_agents), np.zeros(1))

                else:
                    action_p, action_q, action_q_traded = buyers[m]['P_Q_actor'].action(buyers[m]['P_Q_State'])

                    action_p = np.matmul(buyers[m]['Via_Matrix'], action_p)
                    action_q = np.matmul(buyers[m]['Via_Matrix'], action_q) / np.matmul(
                        np.matmul(buyers[m]['Via_Matrix'], action_q), buyers[m]['Customised_Status'])
                    action_q_traded = action_q_traded

                    buyers[m]['A_R_P_Q_Buffer'].store_transition_action_p_q(action_p, action_q, action_q_traded)
                    action_q = -((np.arctanh(action_q_traded) + 18.714973875118524) / 37.42994775023705) * action_q * \
                               buyers[m]['Current_Demand'] + action_q * buyers[m]['Current_Demand']

                    for n in range(len(buyers)):
                        sellers[n]['P_Vector'][buyers[m]['index']] = action_p[sellers[n]['index']]
                        sellers[n]['Q_Vector'][buyers[m]['index']] = action_q[sellers[n]['index']]

            for m in range(len(sellers)):
                if k == (no_of_sub_transactions - 1):
                    sellers[m]['Sub_Reward'] += sellers[m]['Current_Demand'] * lower_price
                    sellers[m]['A_R_P_Q_Buffer'].store_transition_reward(sellers[m]['Sub_Reward'])
                else:
                    sellers[m]['A_R_P_Q_Buffer'].store_transition_reward(0)

            for m in range(len(buyers)):
                if k == (no_of_sub_transactions - 1):
                    buyers[m]['Sub_Reward'] += buyers[m]['Current_Demand'] * grid_price
                    buyers[m]['A_R_P_Q_Buffer'].store_transition_reward(buyers[m]['Sub_Reward'])
                else:
                    buyers[m]['A_R_P_Q_Buffer'].store_transition_reward(0)

        for i in range(no_of_agents):
            agent[i]['Main_Reward'] = agent[i]['Sub_Reward'] + c1 * min(0, agent[i]['Renewable'] + agent[i]['Battery'] -
                                                                        agent[i]['D_S_Action'] - agent[i]['Demand'])
            agent[i]['D_S_Buffer'].store_transition_rewards(agent[i]['Main_Reward'])

        # Training
        if j > 50000:

            for i in range(no_of_agents):
                batch_size = 1

                # Training the D_S Actor Critic first
                current_states, actions, rewards, next_states, next_actions = agent[i]['D_S_Buffer'].sample_buffer(
                    batch_size)
                # Critic Training
                loss_D_S_Critic = agent[i]['D_S_Critic'].train(current_states, actions, next_states, next_actions,
                                                               rewards)
                # Actor Training
                actions = agent[i]['D_S_Actor'].action(current_states)
                grads = agent[i]['D_S_Critic'].gradients(current_states, actions)
                agent[i]['D_S_Actor'].train(current_states, actions, grads)

                # Training the A_R P_Q Actor and A_R_P_Q_Critic
                current_states_a_r, current_states_p_q, actions_a_r, actions_p, actions_q, actions_q_traded, rewards, next_states_a_r, \
                next_states_p_q, next_actions_a_r, next_actions_p, next_actions_q, next_actions_q_traded, via_matrix, customized_status, \
                next_via_matrix, next_customized_status = agent[i]['A_R_P_Q_Buffer'].sample_buffer(batch_size)
                # Critic Training
                loss_A_R_P_Q_Critic = agent[i]['A_R_P_Q_Critic'].train(current_states_a_r, current_states_p_q,
                                                                       actions_a_r, actions_p, actions_q,
                                                                       actions_q_traded, next_states_a_r,
                                                                       next_states_p_q, next_actions_a_r,
                                                                       next_actions_p, next_actions_q,
                                                                       next_actions_q_traded, rewards)
                # Actor Training
                actions_a_r = np.matmul(via_matrix, agent[i]['A_R_Actor'].action(current_states_a_r))
                actions_p, actions_q, actions_q_traded = agent[i]['P_Q_actor'].action(current_states_p_q)
                actions_p = np.matmul(via_matrix, actions_p)
                actions_q = np.matmul(agent[i]['Via_Matrix'], actions_q) / np.matmul(
                    np.matmul(via_matrix, actions_q), customized_status)

                grads_a_r, grads_p, grads_q, grads_q_traded = agent[i]['A_R_P_Q_Critic'].gradients(current_states_a_r,
                                                                                                   current_states_p_q,
                                                                                                   actions_a_r,
                                                                                                   actions_p, actions_q,
                                                                                                   actions_q_traded)
                agent[i]['A_R_Actor'].train(current_states_a_r, actions_a_r, grads_a_r)
                agent[i]['P_Q_ACtor'].train(current_states_p_q, None, grads_p, grads_q, grads_q_traded)
