import pickle
import matplotlib.pyplot as plt

def learning_curve(n):
    fname = "out/losses_vae%i.pt.pkl" % n
    with open(fname, "rb") as f:
        losses = pickle.load(f)
    losses = list(map(float, losses))

    plt.plot(losses)
    plt.xlabel("Epoch number")
    plt.ylabel("VAE loss")
    plt.title("VAE learning curve - %i line programs" % n)

    plt.plot(losses)

    plt.savefig("plots/LC%i.png" % n)
    plt.clf()


for k in range(1, 6):
    learning_curve(k)
