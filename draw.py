# ############## 根据对txt文件 写入、读取数据，绘制曲线图##############
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    fp1 = open('./dist_model/avgloss.txt', 'r')
    total_loss = []
    i = 0
    for loss in fp1:
        if i >= 15:
            loss = loss.strip('\n')  # 将\n去掉
            total_loss.append(loss)
        i += 1
    fp1.close()

    total_loss = np.array(total_loss, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float
    # print(total_loss)

    fp2 = open('./dist_model/testacc.txt', 'r')
    total_acc = []
    i = 0
    for acc in fp2:
        if i >= 15:
            acc = acc.strip('\n')  # 将\n去掉
            total_acc.append(acc)
        i += 1
    fp2.close()

    total_acc = np.array(total_acc, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float
    # print(total_acc)

    X = np.linspace(0, 60, 61)
    # Y1 = total_loss
    Y2 = total_acc

    plt.figure(figsize=(8, 6))  # 定义图的大小
    plt.title("Distill Result")

    plt.xlabel("Train Epoch")
    # plt.ylabel("Train Loss")
    plt.ylabel("Test Acc")
    plt.text(X[-8], Y2[-1], "(" + str(int(X[-1])) + ", " + str(Y2[-1]) + ")")

    # plt.plot(X, Y1)
    plt.plot(X, Y2)
    plt.show()
