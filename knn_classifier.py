import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
class KNNClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.x_train_scaled = None
        self.x_test_scaled = None
        self.y_train = None
        self.y_test = None


    def  train(self,x_train, y_train, x_test, y_test, k=3):
        self.x_train_scaled = self.scaler.fit_transform(x_train)
        self.y_train = y_train
        self.x_test_scaled = self.scaler.transform(x_test) if x_test is not None else None
        self.y_test = y_test
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(self.x_train_scaled,self.y_train)
        return self.model
    
    def predict(self,x=None) :
        if x is not None:
            x_scaled = self.scaler.transform(x)
            return self.model.predict(x_scaled)
        else:
            return self.model.predict(self.x_test_scaled)
        


    def confusion_matrix(self):
        y_pred = self.predict()
        cm = confusion_matrix(self.y_test, y_pred)
        return cm
    
    def accuracy(self) :
        y_pred = self.predict()
        acc = accuracy_score(self.y_test, y_pred)
        print(f"accuracy = {acc:.4f}")
        return acc
    def cross_validate(self, x, y, k=5):
        x_scaled = self.scaler.fit_transform(x)
        results = {}
    
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, x_scaled, y, cv=k, scoring='accuracy')
        results[5] = {'mean:': scores.mean(), 'std': scores.std()}
        print(f"k={5} with Accuracy: {scores.mean():.4f} +- {scores.std():.4f}")
    
        best_k = max(results.keys(), key=lambda x: results[x]['mean'])
        print(f" Best k = {best_k} with accuracy: {results[best_k]['mean']:.4f}")
        
        return results, best_k

