from methods.arc import arc
from methods import hessian
from problems import svd

def main():
    print('Testing...')

    minimum = svd(arc)

    print(f'Minimum objective value: {minumum}')

if __name__ == '__main__':
    main()
