import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=2)


class confusionMatrix:

    def __init__(self, h = 5, w = 5, m = None):
        self.matrix = None
        if m == None:
            self.matrix  = self.getDefaultMatrix(h,w)
        else:
            self.matrix = m
    
    def getDefaultMatrix(self, h,w):
        return [[0 for c in range(w)] for r in range(h)] 

    def setConfusionMatrix(self, input, output):
        print "input: " + str(input) + "output: " + str(output)
        self.matrix[input][output] += 1

    def getTheView(self):
        df_cm = pd.DataFrame(self.matrix, index = [i for i in range(len(self.matrix))],
            columns = [i for i in range(len(self.matrix))])
        plt.figure(figsize = (10,7))
        plt.title('confusion matrix without confusion')
        sns.heatmap(df_cm, annot=True)
        plt.show()

    def toString(self):
        for i in self.matrix:
            print i

def main():
    m = [[5, 0, 0],
        [0, 3, 0],
        [0, 0, 2]]
    cm = confusionMatrix(m = m)
    cm.setConfusionMatrix(1,2)
    cm.toString()
    cm.getTheView()


if __name__ == '__main__':
    main()