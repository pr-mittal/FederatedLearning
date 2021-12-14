import numpy as np
import matplotlib.pyplot as plt
def initializeWeights(num_clients=10):
    # Organization n 2 N randomly initializes γn(0)(rounds), πn(0)(money);
    gamma=[list(np.random.rand(num_clients)*10)]
    pi=[list(np.random.rand(num_clients)*100)]
    utility=10*np.ones(num_clients)
    # t<-0, Convg Indicator<- 0;
    t=0 
    convg_indicator=0
    eta=0.3
    pho=0.00005
    fi=0.0001
    # while Convg Indicator = 0 do
    while convg_indicator==0:
        #Organization n 2 N submits (γn(t); πn(t));
        #   Central server sends organizations γ(t) and π(t) ;
        #   for organization n 2 N in parallel do
        #      γ^n(t) arg maxγn2[0;r¯] Vnρ(γn; γ−n(t); π(t));
        #      γn(t + 1)=γn(t) + η (^ γn(t) − γn(t));
        #      πn(t+1) πn(t)+ρη(γµ(n−2)(t)−γµ(n−1)(t));
        #   end
        gamma_bar=[]
        for i in range(num_clients):
            # γ^n(t) arg maxγn2[0;r¯] Vnρ(γn; γ−n(t); π(t));
            gamma_bar=gamma_bar+[arg_max(gamma[t],pi[t],pho,i,utility[i])]
        gamma=gamma+[list(np.array(gamma[t])+eta*(np.array(gamma_bar) - np.array(gamma[t])))]
        pi=pi+[list(np.array(pi[t])+eta*pho*(np.concatenate([gamma[t][-2:],gamma[t][:-2]]) - np.concatenate([gamma[t][-1:],gamma[t][:-1]])))]
        #   t<- t + 1;
        t=t+1
        #   if |γn(t + 1) − γn(t)} ≤ φ; n 2 N then
        #    Convg Indicator<- 1;
        if(all(i < fi for i in np.abs(np.array(gamma[t]) - np.array(gamma[t-1])))):
            convg_indicator=1
    #  end
    #  end
    #number of rounds
    gamma=np.array(gamma)
    t=[i for i in range(t+1)]
    for i in range(num_clients):
        plt.plot(t,gamma[:,i])
    plt.legend([i for i in range(num_clients)])
    plt.show()
    #cost by organizations
    #ζn= πµ(n+1) − πµ(n+2)
    pi=np.array(pi)
    for i in range(num_clients):
        plt.plot(t,pi[:,(i+1)%num_clients]-pi[:,(i+2)%num_clients])
    plt.legend([i for i in range(num_clients)])
    plt.show()
def arg_max(gamma,pi,pho,i,utility):
    r_bar=np.average(gamma)
    sum=-np.inf
    r_cap=0
    K=5
    T=60
    Dn=0.01
    Sn=600
    e0=9.82
    e1=4.26
    # Model=0.16
    TDL=78.26
    TUL=42.06
    Cninvt=0.22
    CnUL=3
    CnDL=3
    Ccompn=0.174
    for rn in range(1,int(r_bar)+1):
        # rn=gamma[i]
        #e(r(f)) = e0/(e1 + Kr(f));
        #Un(r(f)) = un (e(0) − e(r(f))) ; n in N
        Un=utility*(e0-e0/(e1+K*rn))
        #T=fixed total training time
        #fn*(r(γ))=SnDnK/(T/r(γ) − TnUL − TnDL)
        #T= average time per round * number of rounds
        fn=Sn*Dn*K/(T/rn-TUL-TDL)
        #Cn(fn*,r(f)) = (CnUL + CnDL) r(f) + Cninvtfn* + Ccompn (fn*)^2 Sn Dn Kr(f); n in N
        Cn=(CnUL + CnDL)*rn + Cninvt*fn* + Ccompn *fn*fn*Sn*Dn*K*rn 
        #Ln(rn; λ) = Un(rn) − Cn(fn*(rn), rn)− (πµ(n+2) − πµ(n+1))*rn; n in N:
        Ln=Un-Cn-(pi[(i+2)%len(pi)]-pi[(i+1)%len(pi)])*rn
        #Vρn(γn,γ−n,π) = Vn(fn*(γn), γn, mn(γn1, π))− ρ sum((γµ(n−2) − γµ(n−1)^2) 
        p5=pho*np.sum(np.square(np.concatenate([gamma[-2:],gamma[:-2]]) - np.concatenate([gamma[-1:],gamma[:-1]])))
        #γ^n(t) arg maxγn2[0;r¯] Vnρ(γn; γ−n(t); π(t));
        if(Ln-p5 > sum):
            sum=Ln-p5
            r_cap=rn
    return r_cap
initializeWeights()