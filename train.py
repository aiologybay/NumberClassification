# coding:utf-8
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

n_data = torch.ones(100, 2)  # 100个具有2个属性的数据 shape=(100,2)
x0 = torch.normal(2 * n_data, 1)  # 根据原始数据生成随机数据，第一个参数是均值，第二个是方差，这里设置为1了，shape=(100,2)
y0 = torch.zeros(100)  # 100个0作为第一类数据的标签，shape=(100,1)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # cat数据合并, 0表示按维度行拼接 32-bit floating
y = torch.cat((y0, y1), 0).type(torch.LongTensor)  # 64-bit integer

print(x)
print(y)
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], s=100, lw=0)
# plt.pause(2)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 10, 2)  # 数据是二维的所以输入特征是2，输出是两种类别所以输出层特征是2
print(net)

# plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵 CrossEntropy [0.1, 0.2, 0.7] [0,0,1] 数据越大，是这一类的概率越大

epoch=100
for t in range(1, epoch+1):
    out = net.forward(x)  # 数据经过所有的网络，输出预测值

    loss = loss_func(out, y)  # 输入与预测值之间的误差loss

    optimizer.zero_grad()  # 梯度重置
    loss.backward()  # 损失值反向传播，计算梯度
    optimizer.step()  # 梯度优化
    print('Epoch:{}, Loss:{}'.format(t, loss.data))
    if t % 2 == 0:
        # 画图部分 plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=50, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.5)

# plt.ioff()
plt.show()
