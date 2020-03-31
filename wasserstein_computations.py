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
  
