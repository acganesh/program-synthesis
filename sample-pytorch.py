from model import *
import time
import ast
import os

vae = torch.load('vae-pytorch-50000.pt')
vae.train(False)

TEMPERATURE = 0.01
N_SAMPLES = 10

def random_sample():
    size = vae.encoder.output_size
    rm = Variable(torch.FloatTensor(1, size).normal_())
    rl = Variable(torch.FloatTensor(1, size).normal_())
    if USE_CUDA:
        rm = rm.cuda()
        rl = rl.cuda()
    z = vae.encoder.sample(rm, rl)
    return z

def parses(prog):
    try:
        prog = ast.parse(prog)
        return True
    except:
        return False

N_parse = 0
for i, s in enumerate(range(N_SAMPLES)):
    z0 = random_sample()
    xh = tensor_to_string(vae.decoder.generate(z0,  MAX_LENGTH, TEMPERATURE))
    if xh[-1] == "$":
        xh = xh[:-1]

    print(xh)
    print("==========")

    if parses(xh):
        N_parse += 1
        #print(N_parse)

    #print(i)

print(N_parse)

