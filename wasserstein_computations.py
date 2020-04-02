# Estimating wasserstein distance!
import torch

def sliced_wasserstein(x,y,n_proj=1000):
  '''
  Computes the sliced wasserstein distance between given vectors
  '''
  distances = []
  # random
  rand = torch.randn(x.size(1), n_proj).to(device)  # (slice_size**2*ch)
  rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
  # projection
  proj1 = torch.matmul(x, rand)
  proj2 = torch.matmul(y, rand)
  proj1, _ = torch.sort(proj1, dim=0)
  proj2, _ = torch.sort(proj2, dim=0)
  d = torch.abs(proj1 - proj2)
  return torch.mean(d)
  
def wd1(x, y): return torch.mean(torch.abs(torch.sort(x, dim=0)[0] - torch.sort(y, dim=0)[0]))
  
 
def whitened_wasserstein(y,z):
  # TODO, test depending on if we need it for comparison
  # Implementation using linear independence in the latent space
  # Whitens y, then computes 1D wasserstein, assuming z is spherical gaussian
  
  # Compute the ZCA Matrix
  y1  = y - torch.mean(y,axis=1)
  cov = y1.t() @ y1/(y1.size()[0]-1)

  u,s,_ = torch.svd(cov)
  epsilon = 1e-5
  # ZCA Whitening matrix: U * Lambda * U'
  ZCAMatrix = torch.matmul(u, torch.matmul(torch.diag(1.0/torch.sqrt(s + epsilon)), u.t())) 
  
  #Whiten y
  y_w = ZCAMatrix @ y

  #Compute sum of 1D wassersteins
  proj1, _ = torch.sort(y_w, dim=0)
  proj2, _ = torch.sort(z, dim=0)
  d = torch.abs(proj1 - proj2)
  return torch.sum(d)

def wasserstein_custom(x,y):
  # first dimension is easy since they are independent. 
  x_d1 = x[:, 0]
  y_d1 = y[:, 0]
  w_d1 = wd1(x_d1, y_d1) 

  # now, second dimension we need to handle the inverse of f1. 
  # in particular, we want to compute the inverse on f1^{-1}( f2(x1,x2) )
  # Actually we need the inverse in the following manner -
  # f2(f1^{-1}(x1),x2)
  # In this particular case, it becomes y_d2 - c*x_d1 + (c/(1+a))*x_d1
  x_d2 = x[:, 1]
  y_d2 = y[:, 1]

  # y are encodings, so compute inverse on y 
  assert weights[0,0] != -1
  # y_d2 = y_d2 / (1 + weights[0,0])
  y_d2 = y_d2 - (weights[1,0]/(1 + weights[0,0]))*y_d1 + (weights[1,0]/((1 + weights[0,0])**2))*y_d1 
  w_d2 = wd1(x_d2, y_d2)  
  
  return w_d1 + w_d2, w_d1, w_d2

def transform_cond_dist(y):
  '''
  For y_i = x_i + \sum_{j=0}^{i} w[i,j]*x_j,
  we implement the following transform-
    compute \hat{y}_i s.t. for each j < i, x_j is replaced by f_j^{-1}(x_j), s.t.
  We hope that this gives us the required conditional distribution P(y_i|x_{1..i-1})
  In general, this might need some meta network stuff to achieve R->R, e.g. x1,x2 go into the meta-
  network to determine the weights acting on x3. That way, x3 can be obtained from  
  y3 given x1 and x2.
  '''
  #The algorithm is the following
  #Step 1 - Compute the required x_i which have produced the given y_i 
  #Step 2 - Compute the new x_i^' s.t. applying the function to them would yield x_i
  #Step 3 - Compute the transformed distribution by applying the function to x_i^' and x_i 
  #         s.t. y_i^{o/p} = x_i + weights[i,i]*x_i + \sum weights[i,j] x_j^'

  #step 1
  x = torch.zeros_like(y)
  for i in range(y.size()[1]):
    x[:,i] = (y[:,i] - weights[i,:].view(1,y.size()[1]) @ x.t())/(1+weights[i,i])
  
  #step 2
  x_ = torch.zeros_like(y)
  for i in range(y.size()[1]):
    x_[:,i] = (x[:,i] - weights[i,:].view(1,y.size()[1]) @ x_.t())/(1+weights[i,i])

  #step 3
  y_op = torch.zeros_like(y)
  mask = torch.tril(torch.ones_like(weights),diagonal=-1)   #Since only entries upto i-1 are considered for y_i

  weights_ = weights*mask
  for i in range(y.size()[1]):
    y_op[:,i] = y[:,i] + weights_[i,:].view(1,y.size()[1]) @ ((x_ - x).t())

  return y_op

