import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def fet_red(temp,lim=0):
    """
    temp:training data set
    lim:limit which you want it to be bigger
    RETURNS:new dataset and boolean array
    """
    tem=np.array(temp)
    t=np.sum(tem[:,1:],axis=0)

    x=tem[:,1:]
    y=tem[:,0]
    x_t=np.array(x.T[t>lim]).T
    y=np.reshape(y,(len(y),1))
    n_data=np.concatenate((y,x_t),axis=1)
    return n_data,t>lim


def fet_tra(test_data,arr):
    """
    test_data:array which is to be tested
    arr:Boolean array from fet_red
    RETURNS:modified x and y containing id's,and modified test data

    """
    test_data=np.array(test_data)
    x=test_data[:,1:]
    y=test_data[:,0]
    x=np.array(x.T[arr]).T
    y_=np.reshape(y,(len(y),1))
    n_test=np.concatenate((y_,x),axis=1)
    return x,y,n_test


def man_grph(model,k,s=0):
    """
    model:list of models
    s:from where you want to start
    """
    m=len(model)
    for i in range(m):
        plt.plot(model[i][k],label=f"model {i+s}")
    plt.legend()
    plt.show()    
def grph(J_hist):
    """
    J_hist:array which you want to plot
    
    """
    plt.plot(J_hist)
    plt.show()

def grph_list(model,k):
    """
    model:list of models

    """
    temp=[]
    m=len(model)
    for i in range(m):
        temp.append(model[i][k])
    plt.plot(temp)
    plt.show()
    
def divide(training_data,r):
    """
    training_data:Data which you want to divide
    r:ratio in which you want to divide
    RETURN:x_train,y_train,x_csv,y_csv,train,csv
    """
    training_data=np.array(training_data)
    np.random.shuffle(training_data)
    m=len(training_data)
    Ro=int(m*r)
    train=training_data[:Ro]
    csv=training_data[Ro:]
    x_train=train[:,1:]
    y_train=train[:,0]
    x_csv=csv[:,1:]
    y_csv=csv[:,0]
    return x_train,y_train,x_csv,y_csv,train,csv

def normz(X):
    """
    X:a array with n features and m examples
    RETURN:x_,mean,std_dev
    """
    
    m=len(X)
    # n=len(X[0])
    # x_=np.zeros((m,n))
    mean=np.mean(X,axis=0)
    # mean=1/m*np.sum(X,axis=0)
    std_dev=np.std(X,axis=0)
    # std_dev=(1/m*np.sum((X-mean)**2,axis=0))**0.5
    x_=(X-mean)/std_dev
    # for i in range (m):
    #     for j in range(n):
    #         x_[i][j]=(X[i][j]-mean[j])/std_dev[j]
    return x_,mean,std_dev

def transform(x,mean,std_dev):
    """
    x:array to be transformed
    mean:mean of earlier data
    std_dev:std_dev of earlier data
    RETURN:x_
    """
    # m=len(x)
    # n=len(x[0])
    x_=(x-mean)/std_dev
    # x_=np.zeros((m,n))
    # for i in range (m):
    #     for j in range (n):
    #         x_[i][j]=(x[i][j]-mean[j])/std_dev[j]
    return x_


def R2(y_pred,y):
    """
    y_pred:predicted values
    y:original values
    RETURN:r_2_score
    """
    y_mean=np.mean(y)
    y1=np.sum(np.square(y-y_pred))
    y2=np.sum(np.square(y-y_mean))
    r_2_score=1-(y1/y2)
    return r_2_score


def right(ans1,y):
    """
    ans1:predictions
    y:answers
    RETURN: value
    """
    return np.sum(y==ans1)


def one_hot_encoding(y,no_of_classes):
    """
    y:label array
    no_of_classes: no. of classes in label array
    RETURN:y_one
    """
    m=len(y)
    y_one=np.zeros((m,no_of_classes))
    for i in range (m):
        y_one[i][y[i]]=1

    return y_one

class lin_reg:
    
    


            
    def cost_func (self,X,y,W,b):
        """
        X:a 2d array with n features and m examples
        y:array of output labels of size m
        W:parameter array with size n
        b:parameter scalar
        RETURN:J
        """
        m=len(X)
        # n=len(X[0])
        
        
        J=np.sum((np.dot(X,W)+b-y)**2)#J=scalar x=(m,n) w=(n,) y=(m,)
        J=J/(2*m)    
        return J






    def grad (self,X,y,W,b):
        """
        X:a 2d array with n features and m examples
        y:array of output labels of size m
        W:parameter array with size n
        b:parameter scalar
        RETURN:dj_dw,dj_db
        """
        m=len(X)
        # n=len(X[0])
        # dj_dw=np.zeros(n)
        # dj_db=0
        delta=np.dot(X,W)+b-y#x=(m,n) w=(n,) y=(m,) delta=(m,)
        delta=np.reshape(delta,(len(delta),1))#delta=(m,1)
        dj_dw=np.sum(delta*X,axis=0)#delta=(m,1) x=(m,n) dj_dw=(n,)
        dj_db=np.sum(delta)#delta=(m,1) dj_db=scalar
        # for i in range (m):
            
        #     dj_dw+=(np.dot(X[i],W)+b-y[i])*X[i]
        #     dj_db+=(np.dot(X[i],W)+b-y[i])
        dj_dw=dj_dw/m
        dj_db=dj_db/m
        return dj_dw,dj_db





    def grad_des (self,x_train,y,alpha,epoch,batch_sz=False):
        """
        x_train:a 2d array with n feature and m examples
        y:array of output labels of size m
        w:parameter 
        b:parameter
        alpha:learning rate
        cost_funct1d:to calculate cost
        grad1d:to calculate gradient
        epoch:no.of times you want to run the loop
        RETURN:w,b,J_hist,w_hist,b_hist
        """
        J_hist=[]
        w_hist=[]
        b_hist=[]
        w=np.zeros(len(x_train[0]))
        b=0
        if batch_sz:
            mini_batches=[x_train[k:k+batch_sz] for k in range(0,len(x_train),batch_sz)]
            mini_batches_a=[y[k:k+batch_sz] for k in range(0,len(y),batch_sz)]
            g=0
        else:
            mini_batches=[x_train]
            mini_batches_a=[y]
        for i in range (epoch):    
            for mini,mini_a in zip(mini_batches,mini_batches_a):        

                dj_dw,dj_db=self.grad(mini, mini_a, w, b)
                w=w-alpha*dj_dw
                b=b-alpha*dj_db
                J=self.cost_func(mini,mini_a,w,b)
                J_hist.append(J)
                w_hist.append(w)
                b_hist.append(b)
                
            
            if i%10==0:
                print(f"epoch completed {i} cost is {J_hist[-1]}")    
        print(f"final cost is {J_hist[-1]}") 
        return w,b,J_hist


    def pred(self,x_test,w,b,y_test=np.array([0])):
        """
        x_test:test array
        y_test:labels of test array
        w=parameter
        b=parameter
        RETURN: PREDICTION array
        """
        if y_test.any():
            print(f"cost is {self.cost_func(x_test,y_test,w,b)}")
        return np.dot(x_test,w)+b
    

class pol_reg:
    
    
    def cost_func(self,x,y,w,b,lmbd):
        """
        x:an array with n features and m examples
        y:label array
        w:array of size n with weights
        b:bias
        lmbd:lambda for regularization
        RETURN:J
        """
        
        m=len(x)
        J=np.sum((np.dot(x,w)+b-y)**2)#J=scalar x=(m,n) w=(n,) y=(m,) b=scalar 
        J=J/(2*m)

        
        J+=np.sum(lmbd*(w**2))#lmbd=scalar w=(m,)
        J=J/(2*m)
        return J




    def grad (self,X,y,W,b,lmbd):
        """
        X:a 2d array with n features and m examples
        y:array of output labels of size m
        W:parameter array with size n
        b:parameter scalar
        lmbd:lambda for regularization
        RETURN:J
        """
        m=len(X)
        # n=len(X[0])
        # dj_dw=np.zeros(n)
        # dj_db=0
        delta=np.dot(X,W.T)+b-y#delta=()  x=(m,n)  w=(n,)  b=scalar y=(m,)
        delta=np.reshape(delta,(len(delta),1))#delta=(m,1)
        dj_dw=np.sum(delta*X,axis=0)#dj_dw=(n,) delta=(m,1) x=(m,n)
        dj_dw=dj_dw+lmbd*W#dj_dw=(n,) lmbd=scalar w=(n,)
        dj_db=np.sum(delta)#dj_db=scalar
        # for i in range (m):
        #     for j in range (n):
        #         dj_dw[j]+=(np.dot(x[i],w)+b-y[i])*x[i][j]
        #     dj_db+=(np.dot(x[i],w)+b-y[i])

        # for i in range(len(w)):
        #     dj_dw[i]+=lmbd*w[i]
        dj_dw=dj_dw/m
        dj_db=dj_db/m
        return dj_dw,dj_db




    def polymerise(self,x,degree):
        """
        x:an input array with 3 features and m examples
        degree:no.degrees you want the feature to go
        RETURN: x
        """
        # n=len(x[0])
        m=len(x)

        x1=x[:,0]
        x2=x[:,1]#taking features out
        x3=x[:,2]

        for deg in range(2,degree+1):
            for i in range(deg+1):
                for j in range(deg-i+1):
                    x_t=(x1**i)*(x2**j)*(x3**(deg-i-j))
                    x_t=np.reshape(x_t,(m,1))
                    x=np.concatenate((x,x_t),axis=1)#reshaping in (m,1) and concatenating according to column


        # for i in range(1,degree):
        #     for j in range(n):
        #         x_t=(x[:,j])**(i+1)
        #         x_t=np.reshape(x_t,(m,1))
        #         x=np.concatenate((x,x_t),axis=1)

        # for i in range(2,degree+1):
        #     if i%2==0:
        #         g=int(i/2)
        #         s=0
        #         a=0
        #         h=2
        #         while (g-s-1)>=0:
        #             if(g-s)==(g+a):
        #                 for j in range ((g-1-s)*n,(g-s)*n):
        #                     for k in range(a*n+j+1,(g+a)*n):
        #                         x_t=(x[:,j])*(x[:,k])
        #                         x_t=np.reshape(x_t,(m,1))
        #                         x=np.concatenate((x,x_t),axis=1)
        #             else:
        #                 for j in range ((g-1-s)*n,(g-s)*n):
        #                 for k in range((g+a)*n,(g+1+a)*n):
        #                     if (k-h*n)!=j:
        #                         x_t=(x[:,j])*(x[:,k])
        #                         x_t=np.reshape(x_t,(m,1))
        #                         x=np.concatenate((x,x_t),axis=1)
        #             h+=2            

        #         s+=1
        #         a+=1        
        # else:
        #     g=int(i/2)
        #     s=0
        #     a=0
        #     h=1
        #     while (g-s-1)>=0:
        #         for j in range ((g-1-s)*n,(g-s)*n):
        #             for k in range((g+a)*n,(g+1+a)*n):
        #                 if (k-h*n)!=j:
        #                     x_t=(x[:,j])*(x[:,k])
        #                     x_t=np.reshape(x_t,(m,1))
        #                     x=np.concatenate((x,x_t),axis=1)
        #         a+=1
        #             s+=1
        #             h+=2

        
        # if degree>2:
        #     for i in range(n):
        #         for j in range(i+1,n):
        #             for k in range(j+1,n):
        #                 x_t=(x[:,i])*(x[:,j])*(x[:,k])
        #                 x_t=np.reshape(x_t,(m,1))
        #                 x=np.concatenate((x,x_t),axis=1)
        return x




    def grad_des (self,x_train,y,alpha,epoch,lmbd,batch_sz=False):
        """
        x_train:a 2d array with n feature and m examples
        y:array of output labels of size m
        w:parameter 
        b:parameter
        alpha:learning rate
        cost_funct1d:to calculate cost
        grad1d:to calculate gradient
        epoch:no.of times you want to run the loop
        RETURN:w,b,J_hist
        """
        J_hist=[]
        w_hist=[]
        b_hist=[]
        w=np.zeros(len(x_train[0]))
        b=0
        if batch_sz:
            mini_batches=[x_train[k:k+batch_sz] for k in range(0,len(x_train),batch_sz)]
            mini_batches_a=[y[k:k+batch_sz] for k in range(0,len(y),batch_sz)]
            g=0
        else:
            mini_batches=[x_train]
            mini_batches_a=[y]
        for i in range (epoch):    
            for mini,mini_a in zip(mini_batches,mini_batches_a):        

                dj_dw,dj_db=self.grad(mini, mini_a, w, b,lmbd)
                w=w-alpha*dj_dw
                b=b-alpha*dj_db
                J=self.cost_func(mini,mini_a,w,b,lmbd)
                J_hist.append(J)
                w_hist.append(w)
                b_hist.append(b)
                
            
            if i%10==0:
                print(f"epoch completed {i} cost is {J_hist[-1]}")    
        print(f"final cost is {J_hist[-1]}") 
        return w,b,J_hist



    def pred(self,x_test,w,b,y_test=np.array([0])):
        """
        x_test:test array
        w=parameter
        b=parameter
        RETURN:y_hat
        """
        # m=len(x_test)
        y_hat=np.dot(x_test,w)+b
        if y_test.any():
            print(f"The R2 score is is {R2(y_hat,y_test)}")
        
        
        return y_hat
    
    def pred_test(self,x_csv,model,y_csv=np.array([0]),s=0):
        """
        x_csv:array containing data set
        y_csv:array of output labels
        model:list containing models
        s:from where you started
        RETURN: Alist of predictions
        """
        m=len(model)
        y_hat=[]
        for i in range(m):
            y_hat.append(self.pred(transform(self.polymerise(x_csv,i+s),model[i][-2],model[i][-1]),model[i][1],model[i][2],y_csv))
        return y_hat


    def testing(self,s,st,x,y,alpha,epoch,lmbd):
        """
        s:from degree where we have to start
        st:degree where we need to stop
        x:an array of m examples and n features
        y:an array of size m containings labels
        alpha:learning rate
        ident: number of times you want the loop to learn
        lmbd:regularization parameter
        RETURN:model in manner:y_pred,w,b,lmbd,J_hist,mean,std
        """
        model=[]
        for i in range(s,st+1):
            x_t=self.polymerise(x,i)
            
            x_,mean,std=normz(x_t)
            
            w,b,J_hist=self.grad_des (x_,y,alpha,epoch,lmbd)
            y_pred=self.pred(x_,w,b)
            model.append([y_pred,w,b,lmbd,J_hist,mean,std])
            print(f"cost of exponent {i}={self.cost_func(x_,y,w,b,0),R2(y_pred,y)}")
            
        return model
    

class k_means:
    

    def cost_func(self,x,centroids,idx):
        """
        x:array with n features and m examples
        centroids:array of length k and with n features
        idx:contains index of closest centroid
        RETURN:J 
        """
        m=len(x)
        J=np.sum(np.linalg.norm(x-centroids[idx],axis=1)**2)# x=(m,n)  idx=(m,)

        # for i in range (len(centroids)):
        #     points=x[idx==i]
        #     for j in range (len(points)):
        #         J+=np.linalg.norm(points[j]-centroids[i])
        J=J/m
        return J
        


    def c_cent(self,x,centroids):
        """
        x: array with n features and m examples
        centroids: with n features and k centroids
        RETURN:idx
        """

        x=x.T#x=(m,n)  x.T=(n,m)
        ans=np.matmul(centroids,x)#centroids=(k,n)  x=(n,m)  ans=(k,m)
        mod_x=np.linalg.norm(x,axis=0)#mod_x=(m,)
        mod_query=np.reshape(np.linalg.norm(centroids,axis=1),(len(centroids),1))#mod_query=(k,1)
        mod_x_query=mod_query*mod_x #mod_x_query=(k,m) mod_query=(k,1) mod_x=(m,)
        ans=ans/mod_x_query# ans=(k,m)  mod_x_query=(k,m)
        idx=np.argmax(ans.T,axis=1)#ans.T=(m,k)  idx=(m,)
        
        
        
        # m=len(x)
        # k=len(centroids)
        # idx=np.zeros(m)
        # for i in range (m):
        #     dist=[]
        #     for j in range (k):
        #         l=np.linalg.norm(x[i]-centroids[j])
        #         dist.append(l)
        #     idx[i]=np.argmin(dist)
        return idx



    def comp_cent(self,x,idx,k):
        """
        x:array with n features and m examples
        idx:array of closest index to each examples in x
        k:no. of centroids
        RETURN:centroids
        """
        m,n=x.shape
        centroids=np.zeros((k,n))
        for i in range (k):
            points=x[idx==i]
            centroids[i]=np.mean(points,axis=0)
        return centroids


    def init_cent (self,x,k):
        """
        x:a array with n features and m examples
        k:no. of clusters you want
        RETURN:centroids
        """
        randidx=np.random.permutation(x.shape[0])
        centroids=x[randidx[:k]]
        return centroids



    def k_means(self,x,centroids,iter):
        """
        x:a array with n featyres and m examples
        centroids: array of length of k and n features 
        iter: no of times you want to run the loop
        RETURN:centroids,idx
        """
        k=len(centroids)
        for i in range (iter):
            idx=self.c_cent(x,centroids)
            centroids=self.comp_cent(x,idx,k)
        return centroids,idx    



    def fit (self,x,k,epoch,iter):
        """
        x:array with n features and m examples
        k:no. of cluster you want
        epoch:no. of times you want different value of initial coordinates of centroid
        iter:no. of time you want to run the loop
        RETURN:cent,idx,J_hist[d],cent_hist
        """
        J_hist=[]
        cent_hist=[]
        for i in range (epoch):
            centroids=self.init_cent(x,k)
            l,j=self.k_means(x,centroids,iter)
            cent_hist.append([l,j])
            J_hist.append(self.cost_func(x,l,j))
            
            print(f"ident completed {i}")
        d=np.argmin(J_hist)
        cent=cent_hist[d][0]
        idx=cent_hist[d][1]
        return cent,idx,J_hist[-1],cent_hist
    

    def best_k(self,x,epoch,iter,strt,stp):
        """
        x:training set (m,n)
        epoch:no.of times to initialize the centroids
        iter:no. of times to train the model
        strt:from where you want to start value of k
        stp:where you want to end
        RETURNS: list of all models in manner cent,idx,J_hist[-1],cent_hist 
        """
        model=[]
        for i in range(strt,stp+1):
            model.append(self.fit(x,i,epoch,iter))
            print(f"cost for {i} is {model[-1][2]}")
        return model
    
class KNN:
    def KNN_class(self,x,y,query,k,distance=False):
        """
        x:an array with n features and m examples
        y:an array containing m labels
        query:an array with n features and m queries
        k:no. of closest neighbours you want
        RETURN:q_ans[0,:],if distance is true neighbours
        """
        
        x=x.T #x=(m,n)  x.T=(n,m)
        ans=np.matmul(query,x) #query=(q,n)  x=(n,m)   ans=(q,m)
        mod_x=np.linalg.norm(x,axis=0)#mod_x=(m,)  x=(n,m)
        mod_query=np.reshape(np.linalg.norm(query,axis=1),(len(query),1))#mod_query=(q,1) query=(q,n)
        mod_x_query=mod_query*mod_x#mod_x_query=(q,m)
        ans=ans/mod_x_query#ans=(q,m)  mod_x_query=(q,m)
        # print(np.shape(mod_x_query))
        neighbours=np.argsort(ans,axis=1)#neighbours=(q,m)
        label_k=y[neighbours[:,-k:]]#label_k=(q,k)
        q_ans=pd.DataFrame(label_k.T)
        q_ans=q_ans.mode()
        q_ans=q_ans.to_numpy()
        
    
        
        
        # m=len(x)
        # dist=[]
        # dem=np.shape(query)
        # query=np.reshape(query,(dem[0],1,dem[1]))
        # d=np.linalg.norm(query-x,axis=2)
        # neighbours=np.argsort(d,axis=1)
        # d=np.linalg.norm(x-query,axis=1)
        # neighbours=np.argsort(d)
        # for i in range (m):
        #     d=np.linalg.norm(x[i]-query)
        #     dist.append([d,i])
        # sort_dist=sorted(dist)
        # neighbours=np.array(sort_dist[:k],dtype=np.int64)
        # label_k=y[neighbours[:,:k]]
        # q_ans_=[]
        # q_ans=pd.DataFrame(label_k.T)
        # q_ans=q_ans.mode()
        # # print(q_ans)
        # q_ans=q_ans.to_numpy()
        # q_ans_.append(q_ans[0,:])
        # if(len(q_ans)>1):
        #     print(label_k.T)



        # for i in range (len(query)):
        #     vals,counts=np.unique(label_k[i,:],return_counts=True)
        #     q_ans.append(vals[np.argmax(counts)])

            
        # vals,counts=np.unique(label_k,return_counts=True)
        # mod=np.argmax(counts)
        # q_ans=vals[mod]
        if distance:

            return q_ans[0,:],neighbours
        else:
            return q_ans[0,:]
    




    def fit_class (self,x,y,k,queries,dist=False):
        """
        x:an array with n features and m examples
        y:an array containing m labels
        queries:an array with n features and q query
        k:no. of closest neighbours you want
        RETURN:ans1
        """
        if dist:
            ans1,ans2=self.KNN_class(x,y,queries,k,dist)
            return ans1,ans2
        else:
            ans1=self.KNN_class(x,y,queries,k)
            return ans1
        
       
        
        


    
    

class log_reg:
    def sigm(self,x,w,b):
        """
        x:array with n features and m examples
        w:2d parameter array(classes,features)
        b:parameter array (classes,1)
        RETURN:f_w_b
        """
        # m=len(x)
        # n=len(x[0])
        z=np.dot(w,x)+b#w=(10,784) x=(784,m) b=(10,1) z=(10,m)
        f_w_b=1.0/(1.0+np.exp(-z))#f_w_b=(10,m)
        # print(np.exp(-z))
        # f_w_b=np.zeros(m)
        # for i in range (m):
        #     z=np.dot(x[i],w)+b
        #     f_w_b[i]=1/(1+np.exp(-z))
        return f_w_b        


    def cost_func(self,x,y,w,b):
        """
        x:a array with n features and m examples
        y:output label array
        w:parameter array of size n
        b:parameter
        RETURN:J
        """
        m=len(x[0])
        # n=len(x)
        # J=0
        f_w_b=self.sigm(x,w,b)#f_w_b=(10,m)
        J=np.sum(-y*np.log(f_w_b)-(1-y)*np.log(1-f_w_b))
        # for i in range (m):
        #     J+=-y[i]*np.log(f_w_b[i])-(1-y[i])*np.log(1-f_w_b[i])
        J=J/m
        return J



    def grad (self,x,y,w,b):
        """
        x:a 2d array with n features and m examples
        y:a one hot encoded array
        W:a 2d parameter array (classes,features)
        b:a 2d parameter array (classes,1)
        RETURN:dj_dw,dj_db
        """
        m=len(x[0])
        # n=len(x[0])
        f_w_b=self.sigm(x,w,b)#f_w_b=(10,m)
        # print(f_w_b)
        delta=f_w_b-y#f_w_b=(10,m)  y=(10,m)  delta=(10,m)
        # delta=np.reshape(delta,(m,1))
        dj_dw=np.dot(delta,x.T)#delta=(10,m)  x.T=(m,784) dj_dw=(10,784)
        dj_db=np.reshape(np.sum(delta,axis=1),(len(delta),1))#dj_db=(10,1)
        # dj_dw=np.zeros(n)
        # dj_db=0
        # for i in range (m):
        #     for j in range (n):
        #         dj_dw[j]+=(f_w_b[i]-y[i])*x[i][j]
        #     dj_db+=(f_w_b[i]-y[i])
        dj_dw=dj_dw/m
        dj_db=dj_db/m
        return dj_dw,dj_db




    def log_reg (self,train_data,alpha,ident,no_class):
        """
        train_data:data on which model will be trained
        w:parameter 
        b:parameter
        alpha:learning rate
        cost_funct:to calculate cost
        grad1d:to calculate gradient
        ident:no.of times you want to run the loop
        sigm:to calculate sigmoid
        RETURN:w,b,J_hist
        """
        x=train_data[:,1:785]
        y=train_data[:,0]
        y_one=one_hot_encoding(y,no_class)
        y_one=y_one.T
        x=x.T
        
        J_hist=[]
        # w_hist=[]
        # b_hist=[]
        n=len(x)
        w=np.zeros((len(y_one),n))#w=(classes,features)
        b=np.zeros((len(y_one),1))#b=(classes,1)
        x=x/255
        for i in range (ident):
            dj_dw,dj_db=self.grad(x,y_one,w,b)
            w=w-alpha*dj_dw
            b=b-alpha*dj_db
            J=self.cost_func(x,y_one,w,b)
            J_hist.append(J)
            # w_hist.append(w)
            # b_hist.append(b)
            if i%10==0:
                print(f"ident completed {i} and cost is {J_hist[-1]}")
        print(f"cost in end {J_hist[-1]}")
        return w,b,J_hist




    def pred(self,x,w,b):
        """
        x:data on which to test
        w:parameter array of size n
        b:parameter
        RETURN:y_hat,y
        """
        
        x=x.T
        
        x=x/255
        y_hat=self.sigm(x,w,b)#y_hat=(10,m)
        y_hat=np.argmax(y_hat,axis=0)#y_hat=(m,)
        # for i in range (len(y_hat)):
        #     if y_hat[i]>=0.5:
        #         y_hat[i]=1
        #     else:
        #         y_hat[i]=0    
        return y_hat
    

class NN:




    def Relu(self,z):
        return np.maximum(z,0)
    def leaky_relu(self,z):
        """
        z:array 
        """
        return np.maximum(z,0.01*z)
    
    def leaky_Relu_p(self,z):
        z=np.array(z>0,dtype=np.float32)
        z[z<=0]=0.01
        return z

    def Relu_p(self,z):
        return np.array(z>0,dtype=np.float32)

    def softmax(self,x):
        
        e_x = np.exp(x-np.max(x))
        return (e_x / np.sum(e_x,axis=0))


    


    def cat_cost_funct(self,a,y):
        """
        y:array of desired output
        a:array of activations
        RETURN:J
        """
        J=-np.sum(y*np.log(a))
        return J/len(y)


    def bi_cost_funct(self,a,y):
        """
        y:array of desired output
        a:array of activations
        RETURN:J
        """
        del_=np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))
        J=np.sum(del_)
        return J/len(y)


    def sgmd(self,z):
        """
        z:array of w*x+b
        """
        return 1.0/(1.0+np.exp(-z))

    def sgmd_p(self,z):
        """
        z:array of w*x+b
        """
        return self.sgmd(z)*(1-self.sgmd(z))

    def pred(self,test_data,biases,weights):
        """
        test data:array of the test data
        biases:contains all the biases of the network
        weights:contains all the weights of the network
        RETURN:y_hat,y
        """
        x=test_data[:,1:len(test_data[0])]
        y=test_data[:,0]
        x=x/255
        x=x.T
        for i in range(len(biases)-1):
            x=self.Relu(np.dot(weights[i],x)+biases[i])
            
        x=self.softmax(np.dot(weights[-1],x)+biases[-1])
        # for w,b in zip(weights,biases):
        #     x=sgmd(np.dot(w,x)+b)
        y_fin=np.argmax(x,axis=0)
        return y_fin,y
    def pred_test(self,test_data,biases,weights):
        """
        test data:array of the test data
        biases:contains all the biases of the network
        weights:contains all the weights of the network
        RETURN:y_hat
        """
        test_data=np.array(test_data)
        x=test_data[:,1:]
        
        x=x/255
        x=x.T
        for i in range(len(biases)-1):
            x=self.Relu(np.dot(weights[i],x)+biases[i])
            
        x=self.softmax(np.dot(weights[-1],x)+biases[-1])
        # for w,b in zip(weights,biases):
        #     x=sgmd(np.dot(w,x)+b)
        y_fin=np.argmax(x,axis=0)

        return y_fin

    def back_prop(self,biases,weights,x,y):
        """
        x:array with n features and m examples
        y:output array
        biases:biases of every layer
        weights:weights of every layer
        RETURN:w_grad,b_grad
        """
        b_grad=[np.zeros(b.shape) for b in biases]
        w_grad=[np.zeros(w.shape) for w in weights]
        activation=x
        activations=[x]
        zs=[]
        # print(len(biases))
        for i in range(len(biases)):
            z=np.dot(weights[i],activation)+biases[i]
            # print(z)
            # print(np.shape(z))
            zs.append(z)
            if i<len(biases)-1:
                activation=self.leaky_relu(z)
            else:
                activation=self.softmax(z)
            # print(activation)
            # print(np.shape(activation))
            activations.append(activation)

        
        # for b,w in zip(biases,weights):
        #     z=np.dot(w,activation)+b
        #     # print(z)
        #     zs.append(z)
        #     activation=sgmd(z)
        #     # print(activation)
        #     # print(np.shape(activation))
        #     activations.append(activation)
            
            
        
        delta=(activations[-1]-y)
        delta_temp=np.reshape(np.sum(delta,axis=1),(np.shape(b_grad[-1])))
        b_grad[-1]=delta_temp
        w_grad[-1]=np.dot(delta,activations[-2].transpose())
        for i in range (2,len(weights)+1):
            delta=np.dot(weights[-i+1].transpose(),delta)*self.leaky_Relu_p(zs[-i])
            # print(np.exp(-zs[-i]))
            delta_temp=np.reshape(np.sum(delta,axis=1),(np.shape(b_grad[-i])))
            # print(sgmd_p(zs[-i]))
            
            # print(delta)
            # print(np.shape(delta))
            b_grad[-i]=delta_temp
            w_grad[-i]=np.dot(delta,activations[-i-1].transpose())
            # print(np.shape(delta),np.shape(activations[-i-1].transpose()))
        # delta=(activations[-1]-y)
        # delta_temp=np.reshape(np.sum(delta,axis=1),(np.shape(b_grad[-1])))
        # b_grad[-1]=delta_temp
        # w_grad[-1]=np.dot(delta,activations[-2].transpose())
        # for i in range (2,len(weights)+1):
        #     delta=np.dot(weights[-i+1].transpose(),delta)*sgmd_p(zs[-i])
        #     # print(np.exp(-zs[-i]))
        #     delta_temp=np.reshape(np.sum(delta,axis=1),(np.shape(b_grad[-i])))
        #     # print(sgmd_p(zs[-i]))
            
        #     # print(delta)
        #     # print(np.shape(delta))
        #     b_grad[-i]=delta_temp
        #     w_grad[-i]=np.dot(delta,activations[-i-1].transpose())
        #     # print(np.shape(delta),np.shape(activations[-i-1].transpose()))
        
        return w_grad,b_grad       

    def up_mini(self,biases,weights,mini_batch,alpha,lmbd):
        """
        mini_batch:current mini batch from the training dataset
        biases:biases of every layer
        weights:weights of every layer
        alpha:learning rate
        RETURN:weights,biases
        """
        # nabla_b = [np.zeros(b.shape) for b in biases]
        # nabla_w = [np.zeros(w.shape) for w in weights]
        x=mini_batch[:,1:]
        x=x/255
        y=mini_batch[:,0]
        y_one=one_hot_encoding(y,10)
        x=x.T
        y_=y_one.T
        nabla_w,nabla_b = self.back_prop(biases,weights,x,y_)
        # for x,y in zip(x,y_one):
            
        #     x=np.reshape(x,(784,1))
        #     y=np.reshape(y,(10,1))
        #     delta_nabla_w,delta_nabla_b = back_prop(biases,weights,x, y)
        #     # print(delta_nabla_w)
        #     # print("yo")
        #     nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #     nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        weights=[w-(alpha/len(mini_batch))*wd-(alpha*(lmbd/len(mini_batch)))*w for w,wd in zip(weights,nabla_w)]
        biases=[b-(alpha/len(mini_batch))*bd for b,bd in zip(biases,nabla_b)]
        return weights,biases
        


    def mgd(self,sizes,training_data,mini_batch_sz,alpha,epochs,lmbd,train_data_csv=False):
        """
        x:array of n features m examples
        y:output array
        mini_batch_sz:size of the mini batches
        epochs:no. of times you want to run the algo
        biases:contains all the biases of network
        weights:contains all the weights of the network
        lmbd:lambda for regularization
        train_data_csv:array for cross validation
        RETURN:weights,biases
        """
        biases=[np.random.rand(y,1) for y in sizes[1:]]

        #HE initialization    
        weights=[np.random.uniform(-np.sqrt(6/x),np.sqrt(6/x),(y,x)) for x,y in zip(sizes[:-1],sizes[1:])]    
        
        
        #xavier initialization
        # weights=[np.random.uniform(-np.sqrt(6/(x+y)),np.sqrt(6/(x+y)),(y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
        
        
        # biases=[np.random.rand(y,1) for y in sizes[1:]]
        # weights=[np.random.rand(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1],sizes[1:])]
        accu=[]    
        for j in range (epochs):
            np.random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_sz] for k in range(0,len(training_data),mini_batch_sz)]
            
            for mini_batch in mini_batches:
                weights,biases=self.up_mini(biases,weights,mini_batch,alpha,lmbd)
            
            print(f"epoch:{j}/{epochs}")
            if np.max(train_data_csv):
                y_fin,y_=self.pred(train_data_csv,biases,weights)
                
                acc=np.sum(y_fin==y_)
                print(f"no. of right pred {acc}/{len(train_data_csv)}")
                accu.append((acc/len(train_data_csv))*100)
            # if epochs%10==0:
            #     x_,y_=pred(training_data,biases,weights)
            #     y_=one_hot_encoding(y_,10)
            #     print(cat_cost_funct(x_,y_.T))
        return weights,biases,accu