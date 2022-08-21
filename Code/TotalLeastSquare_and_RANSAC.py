
import numpy as np
import matplotlib.pyplot as plt

xcord_list=[]
ycord_list=[]


"""
Return the center of the red blob in cartesian co-ordinate system


Parameters
----------
 frame: np.array
    Input frame of video
    
Returns
-------
x,y : ints
  co-ordinate of the center in cartesian co-ordinate system.

 """

def extract_data():
    
    """
   extracts the data from csv file
   
   Returns
   -------
   None

    """

    file_path="../data/linear_regression_dataset.csv"
    
    
    data=open(file_path,'r')
    
    rows=data.readlines()
    
    data.close()
    
    #extracting data from .csv file        
    for r in rows[1:]:
        elem_list=r.strip().split(',')
        x=int(elem_list[0])
        xcord_list.append(x)
        y=float(elem_list[6])
        ycord_list.append(y)

#normalizing data
x_scaledlist=[]
y_scaledlist=[]
        
def normalize_data():
    
    """
    normalizes the data from csv file
   
    Returns
    -------
    None

    """
    
    for X in xcord_list:
        x_scaled=(X-min(xcord_list))/(max(xcord_list)-min(xcord_list))
        x_scaledlist.append(x_scaled)
            
    for Y in ycord_list:
        y_scaled=(Y-min(ycord_list))/(max(ycord_list)-min(ycord_list))
        y_scaledlist.append(y_scaled)
    
    #points_array
    
    points=np.array([x_scaledlist,y_scaledlist])
    
    return points


def plot_datapoints():
    
    """
    plots the datapoints and normalized-datapoints 
   
    Returns
    -------
    None

    """

    print("plotting data-points (first-2-plots are data-points plots):\n")
    print('*'*100)
    fig,axes=plt.subplots()
    axes.plot(xcord_list,ycord_list,'ro')
    axes.set_title("Un-scaled datapoints")
    
    fig,axes=plt.subplots()
    #plotting normalized data
    axes.plot(x_scaledlist,y_scaledlist,'bo')
    axes.set_title("normalized datapoints")
    plt.show()               

def covar_matrix():
    
    """
   co-varaince matrix is calculated from which eigen-values are calculated,
   which represent the variance in the data
   
   Returns
   -------
   None

    """

    # Co-variance Matrix
    X_i=points.transpose()
    # print(X_i.shape)
    x=points[0,:]
    y=points[1,:]
    # print(x.shape)
    x_bar=np.mean(x)
    y_bar=np.mean(y)
    x_barlist=[x_bar]*len(x)
    y_barlist=[y_bar]*len(y)
    
    X_bar=np.vstack((x_barlist,y_barlist)).T
    # print(X_bar.shape)
    
    U=np.subtract(X_i,X_bar)
    
    step1=np.dot(U.transpose(),U)
  
    co_var_matrix=(step1/324)
    print('covariance matrix is:\n',co_var_matrix)
    print('+'*50)
    
    #finding eigen_values and eigen_vectors
    print("calculating eigen vectors and eigen values.......")
    evalues,evectors=np.linalg.eig(co_var_matrix)
    print("eigen values are:\n",evalues)
    print('*'*50)
    
    print("1st eigen vector is:")
    evector1=evectors[:,0]
    print(evector1)
    print('*'*50)
    print("2nd eigen vector is:\n")
    evector2=evectors[:,1]
    print(evector2)
    print('*'*50)
    
    origin=[x_bar,y_bar]
    print('plotting data with covariance eigen vectors:\n')
    print('*'*100)
    fig,axes=plt.subplots()
    axes.plot(x_scaledlist,y_scaledlist,'bo')
    axes.set_title("normalized datapoints with co-variance eigen vectors:")
   
    plt.quiver(*origin, *evector1, color=['g'], scale=5,label='λ1 eigenvector')
    plt.quiver(*origin, *evector2, color=['r'], scale=10,label='λ2 eigenvector')
    plt.legend(loc="upper left")
    plt.show()
    



def plot_LScurve():
    
    """
    plots the Least Square Curve
   
    Returns
    -------
    None

    """
  
    #LScurve
    x=points[0]
    # print(x.shape)
    
    Y=points[1]
    # print(Y.shape)
    
    o=np.ones(len(x))
    # print(o.shape)
    
    X=np.vstack((x,o)).T
    # print(X)
    
    step1=np.dot(X.transpose(),X)
    
    step2=np.dot(np.linalg.inv(step1),X.transpose())
    
    B=np.dot(step2,Y)
    
    # print(B)
    
    y_LS_curve=np.dot(X,B)
    
    print("plotting Least Square curve-fitting: ")
    plt.plot(x_scaledlist,y_scaledlist,'bo',x,y_LS_curve,'r-')
    plt.title("Least Square curve-fitting")
    plt.show()

    return y_LS_curve

def plot_TLScurve():
    
    
    """
    plots the  Total Least Square Curve along with Least Square Curve
   
    Returns
    -------
    None

    """
    
# =============================================================================
#     let the line equation be ax+by=c
#     
#     solution of Total Least Square can be found by equation: (U**T)*(U)*(N)=0
#     
#     where U for a line is [x-mean(x),y-mean(y)] and N=[a,b]
# =============================================================================
  
    X_i=points.transpose()

    # print(X_i.shape)
    x=points[0]
    y=points[1]
    
    # print(x.shape)
    x_bar=np.mean(x)
    y_bar=np.mean(y)
    
    U=np.vstack((X_i[:,0]-x_bar,X_i[:,1]-y_bar)).T
    # print(U.shape)
    
    
    # (U**T)*(U) is considered as A
    
# =============================================================================
#     So equation changes to AN=0, which can be solved for N by SVD

#     formula of SVD= U*S*(V**T) , where for any matrix A, U= A*(A**T) , V= (A**T)*A
#     The constant vector (N)  is the eigen-vector corresponding to smallest eigen value
#     in V i.e (A**T)*(A)
# =============================================================================
    
    A=np.dot(U.transpose(),U)
    

    #computing V
    V=np.dot(A.transpose(),A)
    
    #calculating eigen values, and vectors of V
    val,vec=np.linalg.eig(V)
    
    #finding smallest eigen value index
    seigen_val_i=np.argmin(val)
    
    #accessing eigen vector corresponding to smallest eigen vector 
    N=vec[:,seigen_val_i]
    
    # constants a and b
    a,b=N
    
    # calculating constant c
    c=a*x_bar+b*y_bar
   
    y_curve=c-(a*x)
    
    #y co-ordinates of curve
    y_TLS_curve=(y_curve/b)
    # print(y_curve)
    
    plt.plot(x,y,'bo',)
    plt.plot(x,y_TLS_curve,'g-',label='TLS')
    plt.plot(x,y_LS_curve,'r-',label='LS')
    plt.title('Total Least Square Curve Fitting')
    plt.legend(loc="upper left")


def TLS_fit(points):
   
    """
    Fits TLS curve for each 2 points selected , in RANSAC
       
    Returns
    -------
    (a,b,c): tuple
    
    Tuple of integers, which are line-constants
    
    """
    X_i=points.T
    # print(X_i.shape)
    x=points[0]
    y=points[1]
    # print(x.shape)
    x_bar=np.mean(x)
    y_bar=np.mean(y)
    
    U=np.vstack((X_i[:,0]-x_bar,X_i[:,1]-y_bar)).T
    # print(U.shape)
    
    A=np.dot(U.transpose(),U)
    
    N=np.dot(A.transpose(),A)
    
    val,vec=np.linalg.eig(N)
    
    seigen_i=np.argmin(val)
    constants=vec[:,seigen_i]
    
    a,b=constants
    
    c=a*x_bar+b*y_bar
    
    y_curve=c-(a*x)
    y_TLS_curve=(y_curve/b)  
    
    return (a,b,c)


def error(points,r_cons):
   x=points[0]
   y=points[1]
    
   a,b,c=r_cons
    
   E=np.square(a*x+b*y-c)
   
   return E

def Ransac_fit(points,outliers,threshold):
    
    """
    Finds the curve with maximum no of inliers,for the given iterations , in RANSAC
       
    Returns
    -------
    b_const: tuple
    
    tuple of line-constants
    
    """
    
    """
    Note: there is some error in compuation of RANSAC
    """
    #RANSAC
    x=points[0]
    y=points[1]
    
    total_p=points.shape[1]
    
    b_cons=np.zeros((2,1))
    c_points=np.zeros((2,2))
    
    e=outliers/total_p
    s=2
    p=1-e
    N_ipresent=1
    
    count=0
    N=1000
    
    while(N>count):
        #choose a sample
        
        r_indices=np.random.choice(total_p,2)
        
        r_x=x[r_indices]
        r_y=y[r_indices]     
        r_points=np.array([r_x,r_y]).T
        
        r_cons=TLS_fit(r_points)
        
        if np.any(np.iscomplex(r_cons)):
            continue
       
        E=error(points,r_cons)
        
        # print('e is',E)
        
        for i in range(len(E)):
            if float(E[i])>threshold:
                E[i]=0
            else:
                E[i]=1
        
        N_inliers=np.sum(E)
        print("no of inliers is",N_inliers)
    
        if  N_inliers>N_ipresent:
            N_ipresent=N_inliers
            b_cons=r_cons
            c_points=r_points
         
        e=1-(N_inliers)/total_p
        
        p=1-e
          
        N=int(np.log(1-p)/np.log(1-(np.power(1-e,s))))
        
        count=count+1
    return b_cons

def plot_Ransac_curve(const,points):
    
    """
    Plots RANSAC curve
       
    Returns
    -------
    None
    """
    a,b,c=const
    x=points[0,:]
    y=points[1,:]
    
    y_curve=c-(a*x)
    y_Ransac_curve=(y_curve/b)
    
    print('*'*80)
    
    fig,axes=plt.subplots()
    axes.plot(x,y,'bo',x,y_Ransac_curve,'c-')
    axes.set_title('RANSAC curve fitting')
    plt.show()


if __name__=='__main__':
    
    extract_data()
    
    points=normalize_data()
    
    
    plot_datapoints()
        
    covar_matrix()
        
    y_LS_curve=plot_LScurve()
        
    plot_TLScurve()
        
    const=Ransac_fit(points,150,0.1)
        
    plot_Ransac_curve(const,points)

