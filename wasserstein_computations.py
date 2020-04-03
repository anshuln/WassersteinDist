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
  y1  = y - torch.mean(y,axis=0)
  cov = y1.t() @ y1/(y1.size()[0]-1)

  u,s,_ = torch.svd(cov)
  epsilon = 1e-5
  # ZCA Whitening matrix: U * Lambda * U'
  ZCAMatrix = torch.matmul(u, torch.matmul(torch.diag(1.0/torch.sqrt(s + epsilon)), u.t())) 
  
  #Whiten y
  y_w = y @ ZCAMatrix 

  #Compute sum of 1D wassersteins
  proj1, _ = torch.sort(y_w, dim=0)
  proj2, _ = torch.sort(z, dim=0)
  d = torch.abs(proj1 - proj2)
  return torch.mean(d)

def wasserstein_custom(y,z):

  y1_ = transform_cond_dist(y)

  proj1, _ = torch.sort(y1_, dim=0)
  proj2, _ = torch.sort(z, dim=0)
  d = torch.abs(proj1 - proj2)
  
  return torch.sum(d)  

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
    # The masking is done since we need column wise updates, and slice and assign 
    # is not differentiable
    M = torch.zeros_like(x)
    M[:,i] = 1.
    x = x + (((y[:,i] - weights[i,:].view(1,y.size()[1]) @ x.t())/(1+weights[i,i])).view(1,y.size()[0])*M.t()).t()
  
  #step 2
  x_t = torch.zeros_like(y)
  for i in range(y.size()[1]):
    M = torch.zeros_like(x)
    M[:,i] = 1.
    x_t = x_t + (((x[:,i] - weights[i,:].view(1,y.size()[1]) @ x_t.t())/(1+weights[i,i])).view(1,y.size()[0])*M.t()).t()

  #step 3
  y_op = torch.zeros_like(y)
  mask = torch.tril(torch.ones_like(weights),diagonal=-1)   #Since only entries upto i-1 are considered for y_i

  weights_m = weights*mask
  for i in range(y.size()[1]):
    y_op[:,i] = y[:,i] + weights_m[i,:].view(1,y.size()[1]) @ ((x_t - x).t())

  return y_op


