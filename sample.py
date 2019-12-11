from model import *

vae = torch.load('vae-deepcoder-1.pt')
vae.train(False)

TEMPERATURE = 0.01
N_SAMPLES = 1000

def random_sample():
    size = vae.encoder.output_size
    rm = Variable(torch.FloatTensor(1, size).normal_())
    rl = Variable(torch.FloatTensor(1, size).normal_())
    if USE_CUDA:
        rm = rm.cuda()
        rl = rl.cuda()
    z = vae.encoder.sample(rm, rl)
    return z

for s in range(1, N_SAMPLES):
    z0 = random_sample()
    print('(z0)', tensor_to_string(vae.decoder.generate(z0,  MAX_LENGTH, TEMPERATURE)))
    print('\n')

