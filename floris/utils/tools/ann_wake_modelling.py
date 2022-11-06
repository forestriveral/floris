import time
import torch
import import_string
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable
# import multiprocessing as mp
# mp.set_start_method('spawn')


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                     MAIN                                     #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.predict = nn.Linear(n_hidden2, n_output)
    def forward(self, data):
        out = self.hidden1(data)
        out = torch.tanh(out)
        out = self.hidden2(out)
        out = torch.tanh(out)
        out =self.predict(out)
        return out


def net_train(layers, wm="Jensen", num=3000, pshow=False, wsave=False):
    setup_seed(123)

    train_num, test_num = int(num * 0.8), int(num * 0.2)
    wake = "Bastankhah" if wm == "BP" else wm
    data = import_string(f"models.wakes.{wake}.{wm}_data_generator")(num=num)
    # data[:, :-1] = data_normalization(data[:, :-1])
    data = torch.from_numpy(data).cuda()
    torch.set_default_tensor_type(torch.DoubleTensor)
    train_x, train_y = data[:train_num, :-1], data[:train_num, -1][:, None]
    test_x, test_y = data[test_num:, :-1], data[test_num:, -1][:, None]
    # print(train_y)

    device = torch.device("cuda:0")
    n_input, n_hidden1, n_hidden2, n_output = layers
    net = Net(n_input, n_hidden1, n_hidden2, n_output)
    net.to(device).double()
    # print(net)

    epoch = 800
    batch = 2400
    iters = 50
    lr = 0.05

    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=3e-5)
    loss_func = torch.nn.MSELoss()

    torch_dataset = Data.TensorDataset(train_x, train_y)
    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=batch,           # mini batch size
        shuffle=True,               # 设置不随机打乱数据 random shuffle for training
        num_workers=0,              # 使用两个进程提取数据，subprocesses for loading data
    )

    train_errors, test_errors = [], []
    for t in range(epoch):
        errors = []
        for i, (x, y) in enumerate(loader):
            pred = net(x)
            loss = loss_func(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_error = torch.mean(torch.abs(pred - y))
            errors.append(train_error.item())

        train_errors.append(np.mean(errors))
        test_error = torch.mean(torch.abs(net(test_x.cuda()) - test_y.cuda()))
        test_errors.append(test_error.item())

        if (t + 1) % iters == 0:
            print(f"Epoch {t+1} ==> train error: {train_error:.4f} | test error: {test_error:.4f}")

        if t + 1 == epoch:
            print("pred", pred[:20, :])
            print("y", y[:20, :])
            # print("error", y[:20, :].cuda() - pred[:20, :].cuda())

    if pshow:
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(epoch), np.array(train_errors), 'k-', lw=3, label="train error")
        plt.plot(np.arange(epoch), np.array(test_errors), 'b-', lw=3, label="test error")
        plt.xlim((0, epoch + 5))
        plt.legend(loc="upper right"), plt.title("Training curve")
        plt.savefig(f"output/{wm}_train.png", format='png', dpi=300, bbox_inches='tight')
        # plt.show()

    if wsave:
        torch.save(net, f'output/{wm}_net.pkl')


def net_predict(data=None, wm="Jensen", test=False):
    weight = f"output/{wm}_net.pkl"
    net = torch.load(weight, map_location=torch.device("cuda:0"))

    if test:
        wake = wm if wm != "BP" else "Bastankhah"
        test_data = import_string(f"models.wakes.{wake}.{wm}_data_generator")(num=500, test="vect")
        data, label = test_data[:, :-1], test_data[:, -1]
    else:
        assert data is not None

    start = time.time()
    data = torch.from_numpy(data).cuda()
    for i in range(data.shape[0]):
        pred = net(data[i, :])
    end = time.time()
    print(f"net | Using time: {end - start}")

    return pred, label


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 MISCELLANEOUS                                #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def data_normalization(data):
    dmin = np.amin(data, axis=0)
    dmax = np.amax(data, axis=0)

    return (data - dmin) / (dmax - dmin) - 0.5






if __name__ == "__main__":

    layers = (5, 40, 40, 1)

    net_train(layers, wm="BP", pshow=True, wsave=True)
    # pred = net_predict(wm="BP", test=True)
    # print(pred)

