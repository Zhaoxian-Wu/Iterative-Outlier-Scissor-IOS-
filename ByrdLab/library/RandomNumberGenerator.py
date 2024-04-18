import random
import torch

class RngPackage():
    '''
    usage:
    self.random
        performs like build-in lib
        example: generate a random integer number betweem 3-9
            (build-in) 
            res = random.randint(3, 9)
            (RngPack)
            rng_pack = RngPackage(seed=10)
            res = rng_pack.random.randint(3, 9)
    self.torch
        should be set as the parameter 
        of pytorch random number generation function
        example: generate a 5x6 tensor whose elements 
        are random integer numbers betweem 3-9
            (pytorch) 
            res = torch.randint(3, 9, (5, 6))
            (RngPack)
            rng_pack = RngPackage(seed=10)
            res = torch.randint(3, 9, (5, 6)
                                generator=rng_pack.torch)
        
    '''
    def __init__(self, seed=None) -> None:
        self.set_seed(seed)
    def set_seed(self, seed=None):
        if seed is None:
            self.random = random
            self.torch = torch.default_generator
        else:
            self.random = random_rng(seed)
            self.torch = torch_rng(seed)
        
        
def random_rng(seed=10):
    return random.Random(seed)

def torch_rng(seed=10):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
