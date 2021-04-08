from core.sarc_torch .sarc_torch import SARC
from problems.spambase_torch import spambase

if __name__=='__main__':
    spambase(dataroot='problems/spambase', optim_method=SARC, epochs=1, order=2)
    # spambase(dataroot='problems/spambase')
