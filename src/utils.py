
import pandas as pd
import torch

def getSubjectData(subject_filename, columns_names, index_colname,
                   prev_line_string):
    f = open(subject_filename, "rt")
    found = False
    line = f.readline()
    while line is not None and not found:
        if prev_line_string in line:
            found = True
        else:
            line = f.readline()
    subject_data = pd.read_csv(f, names=columns_names)
    subject_data = subject_data.set_index(index_colname)
    return subject_data

def wrap_torch_to_numpy_func(torch_func):
    def scipy_func(x):
        # type(x)==numpy.ndarray
        x_torch = torch.from_numpy(x)
        x_torch.requires_grad = True
        value_torch = torch_func(x_torch)
        value_torch.backward()
        value = value_torch.detach().numpy()
        grad = x_torch.grad.detach().numpy()

        return (value, grad)
    return scipy_func

