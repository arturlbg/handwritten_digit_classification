from collections import Counter

class ClassificationReport:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = sorted(set(y_true).union(set(y_pred)))
        self.tp = sum((self.y_true[i] == label) and (self.y_pred[i] == label) for label in self.labels for i in range(len(self.y_true)))
        self.fp = sum((self.y_true[i] != label) and (self.y_pred[i] == label) for label in self.labels for i in range(len(self.y_true)))
        self.fn = sum((self.y_true[i] == label) and (self.y_pred[i] != label) for label in self.labels for i in range(len(self.y_true)))
        self.tn = sum((self.y_true[i] != label) and (self.y_pred[i] != label) for label in self.labels for i in range(len(self.y_true)))
        self.report_dict = self._generate_report_dict()
        self.precision = self._generate_precision()
        self.recall = self._generate_recall()
        self.f1_score = self._generate_f1_score()
        self.accuracy = '{:.2f}'.format(self._generate_accuracy())
        
    def _generate_report_dict(self):
        report_dict = {}
        for label in self.labels:
            tp = sum((self.y_true[i] == label) and (self.y_pred[i] == label) for i in range(len(self.y_true)))
            fp = sum((self.y_true[i] != label) and (self.y_pred[i] == label) for i in range(len(self.y_true)))
            fn = sum((self.y_true[i] == label) and (self.y_pred[i] != label) for i in range(len(self.y_true)))
            tn = sum((self.y_true[i] != label) and (self.y_pred[i] != label) for i in range(len(self.y_true)))
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            report_dict[label] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": tp + fn
            }
        return report_dict
    
    def _generate_precision(self):
        tp = sum((self.y_true[i] == label) and (self.y_pred[i] == label) for label in self.labels for i in range(len(self.y_true)))
        fp = sum((self.y_true[i] != label) and (self.y_pred[i] == label) for label in self.labels for i in range(len(self.y_true)))
        return tp / (tp + fp) if tp + fp > 0 else 0
        
    def _generate_recall(self):
        tp = sum((self.y_true[i] == label) and (self.y_pred[i] == label) for label in self.labels for i in range(len(self.y_true)))
        fn = sum((self.y_true[i] == label) and (self.y_pred[i] != label) for label in self.labels for i in range(len(self.y_true)))
        return tp / (tp + fn) if tp + fn > 0 else 0
    
    def _generate_f1_score(self):
        precision = self.precision
        recall = self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    def _generate_accuracy(self):
        correct = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == pred)
        return correct / len(self.y_true)
        
    def __str__(self):
        report = "  precision    recall     f1-score    support\n"
        for k, v in self.report_dict.items():
          precision = '{:.2f}'.format(v["precision"])
          recall = round(float(v["recall"]), 2)
          recall = '{:.2f}'.format(v["recall"])
          f1_score = '{:.2f}'.format(v["f1-score"])
          support = '{:.2f}'.format(v["support"])
          report += (f"{k}   {precision}        {recall}        {f1_score}       {support}")
          report+="\n"
        report += "\naccuracy: " + self.accuracy
        report += "\n\nConfusion Matrix: \n             Positive    Negative"
        report += "\nPositive       "+str(self.tp) + "         "+str(self.fn)
        report += "\nNegative       "+str(self.fp) + "         "+str(self.tn)
        return report