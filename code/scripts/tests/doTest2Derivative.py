
import sys
import torch

def main(argv):
    def f(x):
        answer = x[0]**4 + x[1]**3
        return answer
    x = torch.tensor([1, 2], dtype=torch.double, requires_grad=True)
    FAtx = f(x)
    gradFAtx = torch.autograd.grad(FAtx, x, create_graph=True)[0]
    ones = torch.ones_like(x)
    grad2FAtx = torch.autograd.grad(gradFAtx, x, grad_outputs=ones, create_graph=True)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

