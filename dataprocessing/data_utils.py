import torch


class NormalizeTransform:
    def __init__(self, info: dict, out_range = (0, 1)):
        self.info = info
        self.out_min, self.out_max = out_range 

    def __call__(self,data, type = "Inputs"):
        for prop, stats in self.info[type].items():
            index = stats["index"]
            if index < data.shape[0]:
                self.__apply_norm(data,index,stats)
        return data
    
    def reverse(self,data,type = "Labels"):
        for prop, stats in self.info[type].items():
            index = stats["index"]
            self.__reverse_norm(data,index,stats)
        return data
    
    def __apply_norm(self,data,index,stats):
        norm = stats["norm"]
        
        def rescale():
            delta = stats["max"] - stats["min"]
            data[index] = (data[index] - stats["min"]) / delta * (self.out_max - self.out_min) + self.out_min
        
        if norm == "LogRescale":
            data[index] = torch.log(data[index] - stats["min"] + 1)
            rescale()
        elif norm == "Rescale":
            rescale()
        elif norm == "Standardize":
            data[index] = (data[index] - stats["mean"]) / stats["std"]
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{stats['norm']}' not recognized")
        
    def __reverse_norm(self,data,index,stats):
        norm = stats["norm"]

        def rescale():
            delta = stats["max"] - stats["min"]
            data[index] = (data[index] - self.out_min) / (self.out_max - self.out_min) * delta + stats["min"]

        if norm == "LogRescale":
            rescale()
            data[index] = torch.exp(data[index]) + stats["min"] - 1
        elif norm == "Rescale":
            rescale()
        elif norm == "Standardize":
            data[index] = data[index] * stats["std"] + stats["mean"]
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{stats['Norm']}' not recognized")
