#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:20:24 2023

@author: Pierre Weiss, Nathanael Munier, Valentin Debarnot 
(from a code initially developed in Matlab by Emmanuel Soubies)
You can also take a look at this excellent blog
http://kmdouglass.github.io/posts/simulating-microscope-pupil-functions/
"""

__all__ = ['PSFGenerator2Dzernike_t', 'A_op', 'AT_op']

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def define_zernike():
    """
    ------------------------------------------------
    __define_zernike__():
        Return a list of Zernike polynomials lambda functions in Cartesian coordinates
    ------------------------------------------------
    
    Z1:Z00 Piston or Bias
    Z2:Z11 x Tilt
    Z3:Z11 y Tilt
    Z4:Z20 Defocus
    Z5:Z22 Primary Astigmatism at 45
    Z6:Z22 Primary Astigmatism at 0
    Z7:Z31 Primary y Coma
    Z8:Z31 Primary x Coma
    Z9:Z33 y Trefoil
    Z10:Z33 x Trefoil
    Z11:Z40 Primary Spherical
    Z12:Z42 Secondary Astigmatism at 0
    Z13:Z42 Secondary Astigmatism at 45
    Z14:Z44 x Tetrafoil
    Z15:Z44 y Tetrafoil
    Z16:Z51 Secondary x Coma
    Z17:Z51 Secondary y Coma
    Z18:Z53 Secondary x Trefoil
    Z19:Z53 Secondary y Trefoil
    Z20:Z55 x Pentafoil
    Z21:Z55 y Pentafoil
    Z22:Z60 Secondary Spherical
    Z23:Z62 Tertiary Astigmatism at 45
    Z24:Z62 Tertiary Astigmatism at 0
    Z25:Z64 Secondary x Trefoil
    Z26:Z64 Secondary y Trefoil
    Z27:Z66 Hexafoil Y
    Z28:Z66 Hexafoil X
    Z29:Z71 Tertiary y Coma
    Z30:Z71 Tertiary x Coma
    Z31:Z73 Tertiary y Trefoil
    Z32:Z73 Tertiary x Trefoil
    Z33:Z75 Secondary Pentafoil Y
    Z34:Z75 Secondary Pentafoil X
    Z35:Z77 Heptafoil Y
    Z36:Z77 Heptafoil X
    Z37:Z80 Tertiary Spherical
    """
    Z = [None for k in range (38)]
    def r2(x,y): return x**2+y**2
    
    sq3 = 3**0.5
    sq5 = 5**0.5
    sq6 = 6**0.5
    sq7 = 7**0.5
    sq8 = 8**0.5
    sq10 = 10**0.5
    sq12 = 12**0.5
    sq14 = 14**0.5
    
    Z[0] = lambda x,y : torch.ones_like(x) #piston
    Z[1] = lambda x,y : torch.ones_like(x) #piston
    Z[2] = lambda x,y : 2*x #tilt x
    Z[3] = lambda x,y : 2*y #tilt y
    Z[4] = lambda x,y : sq3*(2*r2(x,y)-1) #defocus
    Z[5] = lambda x,y : 2*sq6*x*y 
    Z[6] = lambda x,y : sq6*(x**2-y**2)
    Z[7] = lambda x,y : sq8*y*(3*r2(x,y)-2)
    Z[8] = lambda x,y : sq8*x*(3*r2(x,y)-2)
    Z[9] = lambda x,y : sq8*y*(3*x**2-y**2)
    Z[10] = lambda x,y : sq8*x*(x**2-3*y**2)
    Z[11] = lambda x,y : sq5*(6*r2(x,y)**2-6*r2(x,y)+1)
    Z[12] = lambda x,y : sq10*(x**2-y**2)*(4*r2(x,y)-3)
    Z[13] = lambda x,y : 2*sq10*x*y*(4*r2(x,y)-3)
    Z[14] = lambda x,y : sq10*(r2(x,y)**2-8*x**2*y**2)
    Z[15] = lambda x,y : 4*sq10*x*y*(x**2-y**2)
    Z[16] = lambda x,y : sq12*x*(10*r2(x,y)**2-12*r2(x,y)+3)
    Z[17] = lambda x,y : sq12*y*(10*r2(x,y)**2-12*r2(x,y)+3)
    Z[18] = lambda x,y : sq12*x*(x**2-3*y**2)*(5*r2(x,y)-4)
    Z[19] = lambda x,y : sq12*y*(3*x**2-y**2)*(5*r2(x,y)-4)
    Z[20] = lambda x,y : sq12*x*(16*x**4-20*x**2*r2(x,y)+5*r2(x,y)**2)
    Z[21] = lambda x,y : sq12*y*(16*y**4-20*y**2*r2(x,y)+5*r2(x,y)**2)
    Z[22] = lambda x,y : sq7*(20*r2(x,y)**3-30*r2(x,y)**2+12*r2(x,y)-1)
    Z[23] = lambda x,y : 2*sq14*x*y*(15*r2(x,y)**2-20*r2(x,y)+6)
    Z[24] = lambda x,y : sq14*(x**2-y**2)*(15*r2(x,y)**2-20*r2(x,y)+6)
    Z[25] = lambda x,y : 4*sq14*x*y*(x**2-y**2)*(6*r2(x,y)-5)
    Z[26] = lambda x,y : sq14*(8*x**4-8*x**2*r2(x,y)+r2(x,y)**2)*(6*r2(x,y)-5)
    Z[27] = lambda x,y : sq14*x*y*(32*x**4-32*x**2*r2(x,y)+6*r2(x,y)**2)
    Z[28] = lambda x,y : sq14*(32*x**6-48*x**4*r2(x,y)+18*x**2*r2(x,y)**2-r2(x,y)**3)
    Z[29] = lambda x,y : 4*y*(35*r2(x,y)**3-60*r2(x,y)**2+30*r2(x,y)+10)
    Z[30] = lambda x,y : 4*x*(35*r2(x,y)**3-60*r2(x,y)**2+30*r2(x,y)+10)
    Z[31] = lambda x,y : 4*y*(3*x**2-y**2)*(21*r2(x,y)**2-30*r2(x,y)+10)
    Z[32] = lambda x,y : 4*x*(x**2-3*y**2)*(21*r2(x,y)**2-30*r2(x,y)+10)
    Z[33] = lambda x,y : 4*(7*r2(x,y)-6)*(4*x**2*y*(x**2-y**2)+y*(r2(x,y)**2-8*x**2*y**2))
    Z[34] = lambda x,y : (4*(7*r2(x,y)-6)*(x*(r2(x,y)**2-8*x**2*y**2)-4*x*y**2*(x**2-y**2)))
    Z[35] = lambda x,y : (8*x**2*y*(3*r2(x,y)**2-16*x**2*y**2)+4*y*(x**2-y**2)*(r2(x,y)**2-16*x**2*y**2))
    Z[36] = lambda x,y : (4*x*(x**2-y**2)*(r2(x,y)**2-16*x**2*y**2)-8*x*y**2*(3*r2(x,y)**2-16*x**2*y**2))
    Z[37] = lambda x,y : 3*(70*r2(x,y)**4-140*r2(x,y)**3+90*r2(x,y)**2-20*r2(x,y)+1)
    return Z

"""
    ------------------------------------------------
        __cart2pol__(x, y):
        Cartesian to polar coordinates
    ------------------------------------------------
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
"""
def cart2pol(x_t, y_t):
    rho_t = torch.sqrt(x_t**2 + y_t**2)
    phi_t = torch.arctan2(y_t, x_t)
    return(rho_t, phi_t)


"""
2D generator implemented in pytorch

38: defocus (in nm), 39:cutoff

cutoff = (NA/wavelength)*pixelSize
wavenumber = (nI/wavelength)*pixelSize

"""
class PSFGenerator2Dzernike_t():
    def __init__(
            self, list_param = ["Z4","Z5","Z6","Z7","Z8","Z9","Z10"], psf_size = 17, 
            pixel_size = 100, NA = 1.40, wavelength = 610, nI = 1.5, pupil_size = 256, 
            min_coeff_zernike = [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], 
            max_coeff_zernike = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            device = 'cuda', dtype = torch.float32):
        
        self.device = device
        self.dtype = dtype
        
        self.list_param = list_param    # list of parameters to provide
        self.pixel_size = pixel_size    # size of a pixel in nm
        self.NA = NA                    # numerical aperture
        self.wavelength = wavelength    # wavelength of emitted light
        self.nI = nI                    # refractive index of the immersion medium
        self.psf_size = psf_size        # the generated PSF will be an image of size PSFSize x PSFSize

        self.cutoff_available = "cutoff" in list_param # we check whether we want to different wrt to the cut off frequency
        if self.cutoff_available: print("cutoff not available yet")

        self.min_coeff_zernike = torch.tensor(min_coeff_zernike, device=device, dtype=dtype) # these arrays specify the range of the Zernike coefficients
        self.max_coeff_zernike = torch.tensor(max_coeff_zernike, device=device, dtype=dtype)

        self.Zernike = define_zernike()   # a list of functions making it possible to evaluate Zernike polynomials
        
        self.fc = ((NA/wavelength)*pixel_size) # Cutoff frequency 
        self.kb = ((nI/wavelength)*pixel_size) # wavenumber (used for the propagation in z)
        
        self.pupil_size = max(pupil_size, psf_size)      # Discretization of the Fourier plane, the higher res, the most precise the integral
        lin_t = torch.linspace(-0.5, 0.5, self.pupil_size, device=device, dtype=dtype)
        XX_t, YY_t = torch.meshgrid(lin_t/self.fc, lin_t/self.fc, indexing='ij')       # Fourier plane is discretized on [-0.5,0.5]x[-0.5,0.5]
        self.rho_t, th_t = cart2pol(XX_t, YY_t)              # Cartesian coordinates        
        list_zernike = define_zernike()                 # The list of Zernike polynomial functions
        self.propKer_t = torch.exp(-1j*2*torch.pi*(self.kb**2-self.fc**2*self.rho_t**2 + 0j)**.5)
        # self.propKer_t = torch.exp(-1j*2*torch.pi*(self.kb**2-self.rho_t**2 + 0j)**.5) slight uncertainty between the two lines
        
        # In order to avoid layover in Fourier convolution we need to zero pad and then extract a part of image 
        # computed from pupil_size and psf_size
        from math import ceil, floor
        self.pad_pre = ceil((self.pupil_size-self.psf_size)/2)
        self.pad_post = floor((self.pupil_size-self.psf_size)/2)
        
        map_names = {"Z1":1,"Z2":2,"Z3":3,"Z4":4,"Z5":5,"Z6":6,"Z7":7,"Z8":8,"Z9":9,"Z10":10,"Z11":11,"Z12":12,"Z13":13,"Z14":14,"Z15":15,
                     "Z16":16,"Z17":17,"Z18":18,"Z19":19,"Z20":20,"Z21":21,"Z22":22,"Z23":23,"Z24":24,"Z25":25,"Z26":26,"Z27":27,"Z28":28,
                     "Z29":29,"Z30":30,"Z31":31,"Z32":32,"Z33":33,"Z34":34,"Z35":35,"Z36":36,"Z37":37,
                     "defocus":38,"cutoff":39}
        
        self.index_params = [map_names[x] for x in list_param]                              # a list of indices of the parameters
        self.index_params.sort()  #sorting the list
        self.n_zernike = len([ind for ind in self.index_params if ind <= 38])                                      # the number of Zernike coefficients
        self.Z_t = torch.zeros((self.pupil_size, self.pupil_size, self.n_zernike), device = device, dtype = dtype) # the tensor of Zernike polynomials in the pupil plane
        for k in range(len(self.index_params)):
            if self.index_params[k]<38:
                self.Z_t[:,:,k] = list_zernike[self.index_params[k]](XX_t, YY_t) #defining the k-th Zernike polynomial
            elif self.index_params[k]==38:
                self.Z_t[:,:,k] = self.propKer_t

    def generate_random_coeffs(self):
        if torch.is_complex(self.max_coeff_zernike):
            coeffr_t = torch.rand(len(self.list_param), dtype = self.dtype, device = self.device)*(self.max_coeff_zernike.real - self.min_coeff_zernike.real) + self.min_coeff_zernike.real
            coeffi_t = torch.rand(len(self.list_param), dtype = self.dtype, device = self.device)*(self.max_coeff_zernike.imag - self.min_coeff_zernike.imag) + self.min_coeff_zernike.imag
            return (coeffr_t + 1j*coeffi_t)*torch.sign(coeffr_t[0]) # Because a sign change doesn't change the PSF
        else:
            coeffr_t = torch.rand(len(self.list_param), dtype = self.dtype, device = self.device)*(self.max_coeff_zernike - self.min_coeff_zernike) + self.min_coeff_zernike
            #return coeffr_t*torch.sign(coeffr_t[0]) # Because a sign change doesn't change the PSF
            return coeffr_t

    def generate_psf(self, coeff_t):             
        pupil1_t = self.Z_t @ coeff_t[:self.n_zernike]
        pupil2_t = torch.exp(-2j*torch.pi*pupil1_t)
        if self.cutoff_available:
            indicator_t = (self.rho_t <= coeff_t[-1])
        else:
            indicator_t = (self.rho_t <= 1)
        pupil3_t = pupil2_t * indicator_t
        psf1_t = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(pupil3_t)))
        psf2_t = torch.real(psf1_t*torch.conj(psf1_t))

        psf3_t = psf2_t[self.pad_pre:self.pupil_size-self.pad_post, self.pad_pre:self.pupil_size-self.pad_post]
        psf_t = psf3_t/torch.sum(psf3_t)
        return psf_t

"""
def A_op(x, h):
    2D convolution between a batch of images x and a unique filter h 
    x : nbatch x nchannels x n1 x n2
    h : m1 x m2 or nchannels x m1 x m2
    out : tensor of size nbatch x nchannels x (n1-m1+1) x (n2-m2+1)
    
    We use F.conv2d and return only the valid part of the convolution, 
    so that the image is slightly cropped on the boundaries
    
    The code is simply 
    return F.conv2d(x, h[None, None], padding = 'valid')[0]
"""
def A_op(x_t, h_t):
    if x_t.dim() != 4:
        raise ValueError("Image x_t should be 4-dimensional")
        
    nchan = x_t.shape[1]
    if h_t.dim() == 3:
        if h_t.shape[0] == nchan:
            return F.conv2d(x_t, h_t[None], padding = 'valid', groups = nchan)
        else: 
            raise ValueError("Filter h_t and image x_t must have the same number of channels")
    elif h_t.dim() == 2:
        return F.conv2d(x_t, h_t[None, None].expand(nchan, -1, -1, -1), padding = 'valid', groups = nchan)
    else:
        raise ValueError("Invalid shape for h_t")
    

"""
def AT_op(x, h):
    The adjoint operator of A_op 
    x : nbatch x nchannels x (n1-m1) x (n2-m2)
    h : m1 x m2 or nbatch x m1 x m2
    out : tensor of size nbatch x nchannels x n1 x n2

    The code is simply    
    return F.conv_transpose2d(x, h[None,None], padding = 0)
"""
def AT_op(x_t, h_t):
    if x_t.dim() != 4:
        raise ValueError("Image x_t should be 4-dimensional")
        
    nchan = x_t.shape[1]
    if h_t.dim() == 3:
        if h_t.shape[0] == nchan:
            return F.conv_transpose2d(x_t, h_t[None], padding = 0, groups = nchan)
        else: 
            raise ValueError("Filter h_t and image x_t must have the same number of channels")
    elif h_t.dim() == 2:
        return F.conv_transpose2d(x_t, h_t[None,None].expand(nchan, -1, -1, -1), padding = 0, groups = nchan)
    else:
        raise ValueError("Invalid shape for h_t")
    #return F.conv_transpose2d(x_t, h_t[None,None], padding = 0)

#%% Test code to check the package
if __name__ == '__main__':
    #%% Checking the generation of a single PSF
    device = 'cpu'
    NA = 1.40
    psf_size = 64
    pixel_size = 100
    wavelength = 610
    list_param = ["Z2","Z3","Z4","Z5","Z6","Z7","Z8","Z9","Z10"]
    max_coeff_zernike = [0.2]*len(list_param)
    min_coeff_zernike = [-0.2]*len(list_param)
    psf_generator = PSFGenerator2Dzernike_t(
        psf_size = psf_size, pixel_size = pixel_size, wavelength = wavelength,
        list_param = list_param, min_coeff_zernike = min_coeff_zernike, 
        max_coeff_zernike = max_coeff_zernike, device = device)
    
    coeff_t = psf_generator.generate_random_coeffs()
    h_t = psf_generator.generate_psf(coeff_t)
    
    plt.imshow((h_t**0.5).tolist())
    plt.show()
    
    #%% Checking the operator and its adjoint
    from skimage import data
    dtype = torch.float32
    x = data.cat() #WxHxC
    x = x/x.max()
    x_t = torch.tensor(x, dtype = dtype, device = device)
    x_t = torch.permute(x_t, (2,0,1))[None] # to get 1xCxWxH
    Ax_t = A_op(x_t, h_t)
        
    Ax = torch.permute(Ax_t,(0,2,3,1))[0].detach().cpu() 
    plt.imshow(x);plt.title("Original cat");plt.show()   
    plt.imshow(Ax);plt.title("Blurred cat");plt.show()
    
    y_t = Ax_t[:,[2,0,1],:,:]
    Aty_t = AT_op(y_t, h_t)
    
    print("<Ax,y> = %1.6e" % torch.sum(Ax_t*y_t))
    print("<x,Aty> = %1.6e" % torch.sum(x_t*Aty_t))
    
