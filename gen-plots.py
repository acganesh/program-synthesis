import pickle
import matplotlib.pyplot as plt

def learning_curve(n):
    fname = "out/losses_vae%i.pt.pkl" % n
    with open(fname, "rb") as f:
        losses = pickle.load(f)

    plt.plot(losses)


learning_curve(2)
