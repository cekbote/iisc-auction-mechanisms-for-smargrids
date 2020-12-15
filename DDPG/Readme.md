# Contains the Code for the DDPG Implementation of the Markets in Smartgrids
### Main-Controllers

ADL: Activities of Daily Living

D_S: Demand and Supply
### Sub-Controllers

A_R: Stands for Accept Reject

P_Q: Price and Quantity

A_R_P_Q: Combination of the above

### To Do

Figure out how to constraint the outputs (Can be done with a multiplication in a lamda layer (to only get the relevant outputs) for the sub controllers. DONE

Create a Market Code. DONE

Add discount (gamma factors) to all the actor critics. NEEDS TO BE DISCUSSED

Add a better loss function to make the quantity network make no weird deisions. DONE BUT NEEDS TO BE DISCUSSED

Deciding some parameter Values (Havent Entered them)

Writing the get_renewable and update_adl function and get_demand function.

Incorporating such a loss: Q(s1,s2,a1,a2,a3,a4) - a1'[Loss Values] for the A_R_Network

The constants of the equations have to be discussed. Moreover, the equations have to have if conditionals. DONE
