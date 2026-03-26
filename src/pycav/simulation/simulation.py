import torch
import numpy
from pycav.physics.propagation import compute_steering_vectors


class getPositions:
    @classmethod
    def random_ellipse(cls,x_position_lim = 0e-3, z_position_lim = 20e-3, x_size_lim = 1e-3, z_size_lim = 1e-3, angle = 0, density=100):

#### Calculation of the size of the z axis of the ellipse
        def get_val(lim):
            lim=torch.as_tensor(lim)
            if lim.ndim == 0 or lim.shape[0] == 1:
                return lim
            return lim[0] + torch.rand(1) * (lim[1] - lim[0])
        
        x_size=get_val(x_size_lim)
        z_size = get_val(z_size_lim)
       
#### Calculation of the center of the ellipse (axe z) 
        z_position_lim=torch.as_tensor(z_position_lim)
        if z_position_lim.numel() == 1:    
            z_center=z_position_lim           

        elif z_position_lim.numel() == 2:
            z_center=z_position_lim[0]+z_size/2+torch.rand(1)*(z_position_lim[1]-z_position_lim[0]-2*z_size/2)            
        else:
            raise ValueError("z_position_lim should be of size 1 or 2")

#### Calculation of the center of the ellipse (axe x) 
        x_position_lim=torch.as_tensor(x_position_lim)
        if x_position_lim.shape[0] == 1:    
            x_center=x_position_lim           

        elif x_position_lim.shape[0] == 2:
            x_center=x_position_lim[0]+x_size/2+torch.rand(1)*(x_position_lim[1]-x_position_lim[0]-2*x_size/2)            
        else:
            raise ValueError("z_position_lim should be of size 1 or 2")    

        center = torch.cat([x_center, z_center])
        ellipse_axes = torch.cat([x_size, z_size])
        
        number_bubble = int(torch.round(x_size * z_size * density * 1e6).item())
        
        pos_temp=-0.5 * ellipse_axes+ellipse_axes*torch.rand(number_bubble,2)
        
        ind_in=(pos_temp[:,0]**2/(0.5*x_size)**2+pos_temp[:,1]**2/(0.5*z_size)**2)<=1
        pos_temp2 = pos_temp[ind_in,:]

        if angle == None:
            angle = 360 * torch.rand(1)
        angle_rad = torch.as_tensor(angle * torch.pi/180)
        cos_a = torch.cos(angle_rad,dtype=torch.float64)
        sin_a = torch.sin(angle_rad,dtype=torch.float64)
        
        # Matrice de rotation 2D
        R = torch.tensor([[cos_a, -sin_a],
            [sin_a, cos_a]])
        bubbles_pos = torch.matmul(pos_temp2, R.t()) + center    
        return  bubbles_pos, center, x_size, z_size, angle

class BubbleSource:
    def __init__(self, probe, bubble_positions, c=1540):
        self.probe = probe
        self.c = c
        self.bubble_positions = bubble_positions

class Vokurka(BubbleSource):
    def get_kernel():
        """Cette fonction fourni le kernel de Vokurka"""
        pass

class Stable(BubbleSource):
    def get_kernel():
        """Cette fonction fourni le kernel inertiel"""
        pass
class inertial(BubbleSource):   
    def get_kernel():
        pass

class Simulator:
    def __init__(self,probe,bubble_positions, c=1540,fhifu=1e6, device="cpu"):
        self.probe=probe
        self.bubble_positions=bubble_positions
        self.c=c
        self.fhifu=fhifu
        self.nb = bubble_positions.shape[0]
        self.device = device
class Dirac_Simulator(Simulator):
    def get_delayed_dirac(self, n_cycles,A0):
        # Delay operation
        t_delay=torch.cdist(self.probe.positions,self.bubble_positions,p=2)
        index_tau=t_delay/self.c0*self.probe.fs

        # Create the dirac
        amplitudes = A0 + 0.1*torch.rand(n_cycles,self.nb)
        index_dirac = (torch.arange(n_cycles, device=self.device).float()+0.5) * self.probe.fs/self.fhifu
        t_final = index_tau.unsqueeze(-1) + index_dirac.view(1, 1, -1)


