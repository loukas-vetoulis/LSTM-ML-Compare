import numpy as np

class NaiveBayesBinary:
    def __init__(self, laplace=1.0):  
        self.laplace = laplace
    
    def train(self, X, y): 
        self.types = np.unique(y)  #Ftiaxnei ena sunolo pou periexei kathe timh tou y

        self.typeQuantity = {type: np.mean(y == type) for type in self.types} #Ftiaxnei ena le3iko pou deixnei to P(y) gia kathe diaforetiko typo y.

        self.probs = {}
        for type in self.types:
            X_type = X[y == type] #Epilogh apo ton pinaka twn x mono auta opou einai tou epithimotou typou
            self.probs[type] = (np.sum(X_type, axis=0) + self.laplace) / (X_type.shape[0] + 2 * self.laplace) 
            # Ypologismos meso tou tupou tou bayes ths desmevmenhs pithanotitas kathe lekseis dedomenou toy kathe typou. 
            # to np.sum(X_type, axis=0) dhmiourgei enan pinaka mias grammhs pou exei ws stoixeia tou poses fores emfanizetai h kathe leksh sunolika ola ta keimena autou tou tupou
            # Ara sto probs pairnoume telika enan pinaka pou se kathe thesh i exei thn desmevmenh pithanothta P(PXi = 1 / y = type). O oros laplace prostithetai wste na apofugoume akraies
            # times sth pithanothat (100%).

    def predict_log_prob(self, X):
        log_probs = []
        for type in self.types:
            log_tQuantity = np.log(self.typeQuantity[type]) #Metatroph twn posothtwn se log gia pio grhgorous kai akribeis upologismous
            log_prob = np.sum(X * np.log(self.probs[type]) + (1 - X) * np.log(1 - self.probs[type]), axis=1)
            # Ypologismos mesw tou tupou bayes ths desmevmenhs pithanothtas na emfanziontai oles oi lekseis enos X dedomenou oti to X einai tupou type. 
            # Edw kanoume thn upothesh oti oles oi le3eis emfanizontai ane3arthta h mia apo thn allh. Telika pernoume enan pinaka mias grammhs o opoios 
            # se kathe thesh exei to P(X / Y = type). To log xrhsimopoieitai gia megaluterh akriveia twn upologismwn.
            log_probs.append(log_tQuantity + log_prob)
        return np.array(log_probs).T  # Transpose to have shape (num_samples, num_classes)

    def predict(self, X):
        log_probs = self.predict_log_prob(X)
        return np.argmax(log_probs, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)