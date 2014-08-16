"""
So far, it takes only differentiable functions from all of R^n -> R.
It would be nice if there was a way to restrict the domain to a subset of R^n.
"""

import numpy as np

# :: Num a, b => (Vector a -> b) -> (Vector a -> b) -> Vector a -> Vector a
# (differentiable) functions from R^n to R
def naiveMinimize(f, df, x0, precision):
    step = findLargestStep(f, df, x0)
    x = np.array(x0) # copy array to avoid mutation
    
    while step > precision:
        x = naiveMinimize_(f, df, x, step)
        step /= 4.0

    return x
    

def naiveMinimize_(f, df, x0, step):
    x = np.array(x0) # copy array to avoid mutation
    f_x = f(x)
    dx = step*scaleToUnit(df(x0))
    while f(x-dx) < f_x:
        x -= dx
        f_x = f(x)
        dx = step*scaleToUnit(df(x0))
    return x


def findLargestStep(f, df, x0):
    f_x0 = f(x0)
    unitGradient = scaleToUnit(df(x0))
    step = 1.0

    if f(x0 + step*unitGradient) < f_x0:
        while f(x0 - step*unitGradient) < f_x0:
            step *= 2.0
        step = step/2.0
    else:
        while f(x0 - step*unitGradient) >= f_x0:
            step /= 2.0

    return step


def scaleToUnit(x):
    return x / np.sqrt(np.dot(x, x))
    

def testQuadratic():
    # minimum at [10, 20, -30]^T
    f  = lambda vec:            (vec[0]-10)**2 + (vec[1]-20)**2 + (vec[2]+30)**2
    df = lambda vec: np.array([ 2*(vec[0]-10)  , 2*(vec[1]-20)  , 2*(vec[2]+30)  ])
    min_x = naiveMinimize(f, df, np.zeros((3,)), 1e-9)

    print(min_x)

if __name__ == "__main__":
    testQuadratic()
