from torch import nn
import torch



class SincSquarePDF(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, c):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        You can use ctx.param to save the param of the function 
        """
        ctx.save_for_backward(input)
        ctx.param = c
        #return (c/math.pi) * (torch.sinc(c*input))**2
        return (torch.sinc(c*input))**2
        #return (torch.sin(input)/input)**2

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input,  = ctx.saved_tensors
        c = ctx.param
        #const = 2./ (math.pi*c*(input**3))
        const = 2./ ((c**2)*(input**3))
        term_1 = c * input * torch.sin(c*input) * torch.cos(c*input)
        term_2 = torch.sin(c * input)**2
        return grad_output * torch.nan_to_num(
            const * (term_1 - term_2), nan=0.0, posinf=0.0, neginf=0.0), None
        
class negcos(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, c):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        You can use ctx.param to save the param of the function 
        """
        ctx.save_for_backward(input)
        ctx.param = c
        return c * (1. - torch.cos(input*c)) / ((input*c)**2)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input,  = ctx.saved_tensors
        c = ctx.param
        #const = 2./ (math.pi*c*(input**3))

        htheta = c * (1. - torch.cos(input*c)) / (torch.pi* ((input*c)**2) )
        const = c/(input**3)
        term1 = input*torch.sin(c*input)
        term2 = 2.0 + 2.0 * torch.cos(c*input)
        #return grad_output * torch.nan_to_num( c/(input*torch.pi) * ( (torch.sinc(c*input)) - 2.0*htheta*torch.pi/c ) , nan=0.0, posinf=0.0, neginf=0.0), None
        return grad_output * torch.nan_to_num( const * ( term1 - term2 ) , nan=0.0, posinf=0.0, neginf=0.0), None

class SinFouthPower(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, c):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used  
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        You can use ctx.param to save the param of the function 
        """
        ctx.save_for_backward(input)
        ctx.param = c
        return 2*c*(torch.sin(c*input)**4) / (torch.pi*((c*input)**2))
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input,  = ctx.saved_tensors
        c = ctx.param
        #const = 2./ (math.pi*c*(input**3))

        htheta = 2.0*c*(torch.sin(c*input)**4) / (torch.pi*((c*input)**2))
    
        return grad_output * torch.nan_to_num(
            (4.0*torch.sin(2.0*c*input)/torch.pi*input- 2*htheta), nan=0.0, posinf=0.0, neginf=0.0), None


def Gaussian_reg(wgts, lambda_):
    return (lambda_ * torch.sum(wgts**2))
        
def Laplacian_reg(wgts, lambda_):
    return (lambda_ * torch.sum(wgts.abs()))

def Cauchy_reg(wgts, lambda_,):
    return (lambda_ * torch.sum((torch.log(1 + (wgts)**2)) ))

def Sinc_squared_reg(wgts, lambda_,c=1):
    sinc_square_pdf = SincSquarePDF.apply
    return (lambda_*-1.0*torch.sum(torch.log(sinc_square_pdf(wgts, c))))

def negcos_reg(wgts,lambda_,c=1):
    negcos_pdf = negcos.apply
    return (lambda_*-1.0*torch.sum(torch.nan_to_num(torch.log(negcos_pdf(wgts, c)), nan=0.0, posinf=0.0, neginf=0.0)))

def SinFouthPower_reg(wgts,lambda_,c=1):
    SinFouthPower_pdf = SinFouthPower.apply
    return (lambda_*-1.0*torch.sum(torch.nan_to_num(torch.log(SinFouthPower_pdf(wgts, c)), nan=0.0, posinf=0.0, neginf=0.0)))

def compute_regularizer_term(wgts, lambda_, hidden_prior, hidden_prior2='False', hyperatio = 1.,c=1):
    term2_ratio = 1 - hyperatio
    if hidden_prior == "Uniform":
        reg_term = 0
    elif hidden_prior == "Gaussian":
        reg_term = hyperatio * Gaussian_reg(wgts, lambda_)
    elif hidden_prior == "Laplace":
        reg_term = hyperatio * Laplacian_reg(wgts, lambda_)
    elif hidden_prior == "Cauchy":
        reg_term = hyperatio * Cauchy_reg(wgts, lambda_)  
    elif hidden_prior == "Sinc_squared":
        reg_term = hyperatio * Sinc_squared_reg(wgts, lambda_,c)
    elif hidden_prior == "negcos":
        reg_term = hyperatio * negcos_reg(wgts, lambda_,c)
    elif hidden_prior == "SinFouthPower":
        reg_term = hyperatio * SinFouthPower_reg(wgts, lambda_,c)   

    if hidden_prior2 == 'False':
        reg_term2=0
    if hidden_prior2 == "Gaussian":
        reg_term2 = term2_ratio * Gaussian_reg(wgts, lambda_)
    elif hidden_prior2 == "Laplace":
        reg_term2 = term2_ratio * Laplacian_reg(wgts, lambda_)
    elif hidden_prior2 == "Cauchy":
        reg_term2 = term2_ratio * Cauchy_reg(wgts, lambda_)  
    elif hidden_prior2 == "Sinc_squared":
        reg_term2 = term2_ratio * Sinc_squared_reg(wgts, lambda_,c)
    elif hidden_prior2 == "negcos":
        reg_term2 = term2_ratio * negcos_reg(wgts, lambda_,c)
    elif hidden_prior2 == "SinFouthPower":
        reg_term2 = term2_ratio * SinFouthPower_reg(wgts, lambda_,c)         
    return reg_term+reg_term2

