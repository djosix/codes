
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import pickle

def save_test1_fig(name):
    with open("test1/{}.pkl".format(name), "rb") as f:
        results = pickle.load(f)
        fps, acc, var = [], [], []
        for r in results:
            fps.append(r["fps"])
            acc.append(r["accuracy"])
            var.append(r["config"][name])

        plt.hold(True)
        plt.plot(var, fps, color="blue")
        plt.plot(var, acc, color="red")
        plt.legend(handles=[
            ptc.Patch(color='blue', label='FPS'),
            ptc.Patch(color='red', label='Accuracy')
        ])
        plt.xlabel(name)
        plt.savefig("test1/{}.png".format(name))
        plt.hold(False)
        plt.close()

save_test1_fig("rescale")
save_test1_fig("step")
save_test1_fig("search")
save_test1_fig("P")
save_test1_fig("Q")
save_test1_fig("sv_max")