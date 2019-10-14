
#Extract model params as strings
def extract_ops_as_string(model):
    params = model.main
    m_params = list(map(lambda x: x.__str__(),params))
    return m_params

# attain index of Convolution operations
def conv_idx(model):
    params = extract_ops_as_string(model)
    idxs = np.nonzero(list(map(lambda x: x.__contains__("Conv"), params)))[0]
    return idxs



def extract_convs_as_str(model):
    idxs = conv_idx(model) # attain index of children that contain convolutions when print(model)
    conv_params = np.array(list(model.children()))[0][idxs] # extract children parameters with "Conv"
    return conv_params

# Partitions Conv grads into bottom and top to track gradient scalarization.
def bottom_top_grads(model):
    conv_params = extract_convs_as_str(model)

    grads = list(map(lambda x: x.weight.grad,conv_params)) # extract gradients
    if grads[0] is None:
        print("Model has NO model.backward() history and thus an input needs to be computed")
        return

    grads = list(map(lambda x: x.view(-1),grads)) # turn gradients into 1-dimension

    half = len(conv_params) // 2
    bottom = grads[:half]
    top = grads[half:]

    bottom_grad_mean = torch.cat(bottom).mean().item()
    top_grad_mean = torch.cat(top).mean().item()
    return [bottom_grad_mean,top_grad_mean] # return bottom and top gradient means

