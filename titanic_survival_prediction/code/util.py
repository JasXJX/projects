import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split as spl
from sklearn import preprocessing as pre
# import numpy as np
import pandas

# devc = "cuda" if torch.cuda.is_available() else "cpu"
devc = "cpu"
torch.device(devc)
torch.manual_seed(3)


def data_loader(path: str) -> pandas.DataFrame:
    with open(path, "r") as f:
        train_source = pandas.read_csv(f, index_col="PassengerId")
    train_source = train_source.loc[:,
                                    ["Survived",
                                     "Pclass",
                                     "Sex",
                                     "Age",
                                     "SibSp",
                                     "Parch",
                                     "Fare"
                                     ]
                                    ]
    train_source["Sex"] = train_source["Sex"].replace(["male", "female"],
                                                      [1, 0]
                                                      )
    train_source["Age"] = train_source["Age"].fillna(0)
    return train_source


def spl_and_norm(source: pandas.DataFrame) -> tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor
                                                    ]:
    train, valid = spl(source, test_size=0.2, random_state=3)
    train_x, train_y = train.drop(labels="Survived", axis=1).to_numpy(), \
        torch.as_tensor(train.loc[:, "Survived"].to_numpy(),
                        dtype=torch.long,
                        device=devc
                        )
    valid_x, valid_y = valid.drop(labels="Survived", axis=1).to_numpy(), \
        torch.as_tensor(valid.loc[:, "Survived"].to_numpy(),
                        dtype=torch.long,
                        device=devc
                        )
    train_x = torch.as_tensor(pre.normalize(train_x, norm="max", axis=0),
                              dtype=torch.float,
                              device=devc
                              )
    valid_x = torch.as_tensor(pre.normalize(valid_x, norm="max", axis=0),
                              dtype=torch.float,
                              device=devc
                              )
    return train_x, train_y, valid_x, valid_y


class TitanicPred(nn.Module):
    def __init__(self):
        super(TitanicPred, self).__init__()
        self.stack = nn.Sequential(nn.Linear(in_features=6,
                                             out_features=24,
                                             device=devc),
                                   nn.ReLU(),
                                   nn.Linear(in_features=24,
                                             out_features=24,
                                             device=devc),
                                   nn.ReLU(),
                                   nn.Linear(in_features=24,
                                             out_features=24,
                                             device=devc),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(in_features=24,
                                             out_features=6,
                                             device=devc),
                                   nn.ReLU(),
                                   nn.Linear(in_features=6,
                                             out_features=2,
                                             device=devc),
                                   nn.Sigmoid()
                                   )

    def forward(self, x):
        return self.stack(x)
