import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def projection(g, f):
    g[f == 0] = np.ma.array(g, mask=f).mean()
    g[f == 1] = np.ma.array(g, mask=np.where(f==1, 0, 1)).mean()
    return g

def E(g):
    return np.ones(g.shape)*np.mean(g)

def argmin(g):
    global images
    values = []
    for i in range(0, 10):
        pr_i_g = projection(np.copy(g), np.copy(images[i]))
        values.append(np.linalg.norm(g - pr_i_g)/np.linalg.norm(pr_i_g - E(g)))
    return np.argmin(values)

if __name__ == '__main__':

    images = []
    for i in range(10):
        img = plt.imread(f'c_{i}.png')
        img[img != 0] = 1.
        images.append(img)

    sigmas = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 20]
    freqs = []
    matrixes = []

    for sigma in sigmas:
        correct_preds = 0
        matrix = np.zeros((10, 10))
        for i in range(10):
            for _ in range(100):
                img_noice = images[i] + np.random.normal(0, sigma, size=(28,28))
                predict = argmin(img_noice)
                if predict == i:
                    correct_preds += 1
                matrix[i, predict] += 1
                
        matrixes.append(matrix)
        freqs.append(1-correct_preds/1000)


    figure, axis = plt.subplots(1, 2)
    
    axis[0].set_xlabel('Сдандартное отклонение')
    axis[0].set_ylabel('Частота ошибок')
    axis[0].set_title('Зависимость частоты ошибок от стандартного отклонения')
    axis[0].grid(True, linestyle='dotted')
    axis[0].set_ylim(0, 1)
    axis[0].set_yticks(np.arange(0, 1, 0.1))
    axis[0].plot(sigmas, freqs, linewidth=2)
    
    sns.heatmap(matrixes[7], annot=True, fmt='g', ax=axis[1])
    axis[1].set_xlabel('Прогноз')
    axis[1].set_ylabel('Истина')
    axis[1].set_title('Матрица ошибок')

    plt.show()