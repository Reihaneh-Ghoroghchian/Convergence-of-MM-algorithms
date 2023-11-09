def channel_random_initialize(n, g, u):
    H=[]
    np.random.seed(1)
    for i in range(g):
        temp=[]
        for j in range(u):
            hik = np.random.rayleigh(1, N)
            temp.append(hik)
        H.append(temp)
    return H


def construct_problem_DC(h,n,g,k,y,w0,s):
    #print("g:",g)
    #print("n:",n)
    w = cp.Variable((g,n))
    objective = cp.Minimize(cp.sum_squares(w))
    #print(cp.sum_squares(w))
    constraints=[]
    for i in range(g):
        #print("in first for g:", i)
        for j in range(k):
            #print("in second for k:",j)
            withik=np.matmul(w0[i],h[i][j])
            #print("2*withik*h[i][j]@(w[i]-w0[i])",2*withik*h[i][j]@(w[i]-w0[i]))
            flag=False
            for b in range(g):
                if b!=i:
                    if flag==False:
                        temp=cp.sum_squares(w[b,:]@h[i][j])
                        flag=True
                    else:
                        temp+=cp.sum_squares(w[b,:]@h[i][j])

            #print("temp:",temp)
            constraints+=[ y[i][j]*temp+y[i][j]*pow(s,2)-np.power(np.matmul(w0[i],h[i][j]),2)-2*withik*np.transpose(h[i][j])@(w[i]-w0[i])<=0]
    prob=cp.Problem(objective, constraints)
    res=prob.solve()

    return w.value


def construct_problem_taylor(h,n,g,k,y,w0,s):
    #print("g:",g)
    #print("n:",n)
    w = cp.Variable((g,n))
    objective = cp.Minimize(cp.sum_squares(w))
    #print(cp.sum_squares(w))
    constraints=[]
    for i in range(g):
        #print("in first for g:", i)
        for j in range(k):
            #print("in second for k:",j)
            first=0
            # first line
            for b in range(g):
                if b!=i:
                    first+=cp.sum_squares(w0[b]@h[i][j])*y[i][j]
            first+=y[i][j]*pow(s,2)
            first-=cp.sum_squares(w0[i]@h[i][j])
            #print(first)
            #second line
            second=0
            for b in range(g):
                if b!=i:
                    second+=np.matmul(w0[b],h[i][j])*y[i][j]*2*np.transpose(h[i][j])@(w[b].T-w0[b].T)
            #print(second)
            second=second-2*np.transpose(h[i][j])*np.matmul(w0[i],h[i][j])@(w[i].T-w0[i].T)
            #print(second)
            #third line
            #print("third")
            #third=(y[i][j]*(number_of_groups-1)-1)*np.sum(h[i][j]**2)
            third=0
            for b in range(g):
                if b!=i:
                    third+=np.sum(h[i][j]**2)*y[i][j]*cp.sum_squares(w[b]-w0[b])
            #print(third)
            #print(np.sum(h[i][j]**2))
            third=third+np.sum(h[i][j]**2)*cp.sum_squares(w[i]-w0[i])#how? why?
            #print(third)
            constraints+=[first+second+third<=0]
    prob=cp.Problem(objective, constraints)
    res=prob.solve()

    return w.value



def construct_problem_nestrov(h,n,g,k,y,w0,s):
    #print("g:",g)
    #print("n:",n)
    w = cp.Variable((g,n))
    objective = cp.Minimize(cp.sum_squares(w))
    #print(cp.sum_squares(w))
    constraints=[]
    for i in range(g):
        #print("in first for g:", i)
        for j in range(k):
            #print("in second for k:",j)
            first=0
            # first line
            for b in range(g):
                if b!=i:
                    first+=cp.sum_squares(w0[b]@h[i][j])*y[i][j]
            first+=y[i][j]*pow(s,2)
            first-=cp.sum_squares(w0[i]@h[i][j])
            #print(first)
            #second line
            second=0
            for b in range(g):
                if b!=i:
                    second+=np.matmul(w0[b],h[i][j])*y[i][j]*2*np.transpose(h[i][j])@(w[b].T-w0[b].T)
            #print(second)
            second=second-2*np.transpose(h[i][j])*np.matmul(w0[i],h[i][j])@(w[i].T-w0[i].T)
            #print(second)

            d=0.1
            constraints+=[first+second+(d/2)*cp.sum_squares(w-w0)<=0]
    prob=cp.Problem(objective, constraints)
    res=prob.solve()

    return w.value




def construct_problem(h,n,g,k,y,w0_w,s,lambda0):
    w = cp.Variable((g,n))
    ob=cp.sum_squares(w)
    constraints=[]
    for i in range(g):
        for j in range(k):
            withik_w=np.power(np.matmul(w0_w[i],h[i][j]),2)
            #print("withik_w",withik_w)
            flag=False
            for b in range(g):
                if b!=i:
                    if flag==False:
                        temp=cp.sum_squares(w[b,:]@h[i][j])
                        flag=True
                    else:
                        temp+=cp.sum_squares(w[b,:]@h[i][j])

            ob+=((y[i][j]*temp+y[i][j]*pow(s,2)+np.power(np.matmul(w0_w[i],h[i][j]),2)-2*np.matmul(w0_w[i],h[i][j])*(w[i]@h[i][j]))*lambda0[i][j])
            constraints+=[(y[i][j]*temp+y[i][j]*pow(s,2)+np.power(np.matmul(w0_w[i],h[i][j]),2)-2*np.matmul(w0_w[i],h[i][j])*(w[i]@h[i][j]))<=0]
    objective = cp.Minimize(ob)
    prob=cp.Problem(objective,constraints)
    res=prob.solve()

    return w.value

def construct_problem_Lambda(h,n,g,k,y,w0_l,s):
    Lambda = cp.Variable((g,k))
    cons=[]
    ob=np.sum(w0_l**2)
    for i in range(g):
        for j in range(k):
            withik=np.power(np.matmul(w0_l[i],h[i][j]),2)
            flag=False
            for b in range(g):
                if b!=i:
                    if flag==False:
                        temp=np.power(np.matmul(w0_l[b],h[i][j]),2)
                        flag=True
                    else:
                        temp+=np.power(np.matmul(w0_l[b],h[i][j]),2)

            #print((y[i][j]*temp+y[i][j]*pow(s,2)-withik))
            ob+=((y[i][j]*temp+y[i][j]*pow(s,2)-withik)*Lambda[i][j])
            cons+=[Lambda[i][j]>=0]

    
    objective = cp.Maximize(ob)
    prob=cp.Problem(objective,cons)
    res=prob.solve()

    return Lambda.value














import numpy as np
import cvxpy as cp
from numpy import linalg as LA
import matplotlib.pyplot as plt

diff_thresh=0.01
number_of_groups = 8
number_of_users_per_groug = 4
N=2
s=1.0
y = np.ones((number_of_groups,number_of_users_per_groug)) # close to zero 
y*=0.01
w0_main = np.ones((number_of_groups,N))
w0_main*=5



#################################DC
w0=w0_main
h =channel_random_initialize(N, number_of_groups, number_of_users_per_groug)
print("y:",y)
print("h:",h)
print("w0:",w0)
counter=0
diff=10
output_dc=[]
output_dc.append(np.sum(w0**2))
step_time_dc=[]
while diff>diff_thresh:
    #print(counter)
    counter+=1
    w0_new = construct_problem_DC(h,N,number_of_groups,number_of_users_per_groug,y,w0,s)# Construct the problem.
    #print("                                            sum:",np.sum(w0_new**2))
    output_dc.append(np.sum(w0_new**2))
    #print(w0_new)
    #print(w0_new)
    diff=LA.norm(w0_new-w0)
    #print("diff",diff)
    w0=w0_new

##################################Taylor
w0 = w0_main
h =channel_random_initialize(N, number_of_groups, number_of_users_per_groug)
print("y:",y)
print("h:",h)
print("w0:",w0)
counter=0
diff=10
output_taylor=[]
output_taylor.append(np.sum(w0**2))
step_time_tay=[]
while diff>diff_thresh:
    #print(counter)
    counter+=1
    w0_new = construct_problem_taylor(h,N,number_of_groups,number_of_users_per_groug,y,w0,s)# Construct the problem.
    #print("                                            sum:",np.sum(w0_new**2))
    output_taylor.append(np.sum(w0_new**2))
    #print(w0_new)
    #print(w0_new)
    diff=LA.norm(w0_new-w0)
    #print("diff",diff)
    w0=w0_new

########################################Nestrov
w0 = w0_main
h =channel_random_initialize(N, number_of_groups, number_of_users_per_groug)
counter=0
diff=10
output_nestrov=[]
output_nestrov.append(np.sum(w0**2))
step_time_nes=[]
while diff>diff_thresh:
    #print(counter)
    counter+=1
    w0_new = construct_problem_nestrov(h,N,number_of_groups,number_of_users_per_groug,y,w0,s)# Construct the problem.
    #print("                                            sum:",np.sum(w0_new**2))
    output_nestrov.append(np.sum(w0_new**2))
    #print(w0_new)
    #print(w0_new)
    diff=LA.norm(w0_new-w0)
    #print("diff",diff)
    w0=w0_new

##############################Lagrange
w0 = w0_main
h =channel_random_initialize(N, number_of_groups, number_of_users_per_groug)
counter=0
diff=10
output_lag=[]
output_lag.append(np.sum(w0**2))
step_time_lag=[]
while diff>diff_thresh:
    #print(counter)
    counter+=1
    lambda0= construct_problem_Lambda(h,N,number_of_groups,number_of_users_per_groug,y,w0,s)
    for i in range(number_of_groups):
        for j in range(number_of_users_per_groug):
            lambda0[i][j]=abs(lambda0[i][j])
    w0_new = construct_problem(h,N,number_of_groups,number_of_users_per_groug,y,w0,s,lambda0)# Construct the problem.
    output_lag.append(np.sum(w0_new**2))
    diff=LA.norm(w0_new-w0)
    w0=w0_new






print(output_dc)
print(output_taylor)
print(output_nestrov)
print(output_lag)

plt.plot(output_dc)
plt.plot(output_taylor)
plt.plot(output_nestrov)
plt.plot(output_lag)
plt.title("G=8, K=4, N=2")
plt.xlabel("Step")
plt.ylabel("Objective Function")
plt.legend(["DC", "Taylor","Nestrov","lag"])
plt.show()
'''
print(output_dc)
plt.plot(step_time_dc)
plt.plot(step_time_tay)
plt.plot(step_time_nes)
plt.plot(step_time_lag)

plt.legend(["DC", "Taylor","Nestrov","lag"])
plt.show()
'''
