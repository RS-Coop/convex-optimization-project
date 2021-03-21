from core.arc.arc import arc
from core import hessian
from problems.svd import svd

def main():
    print('Testing...')

    minimum = svd(arc, {'eps_g':1e-3, 'eps_h':1e-3})

    print(f'Minimum objective value: {minimum}')

if __name__ == '__main__':
    main()
