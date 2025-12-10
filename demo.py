
def sliceDemo():
    a = [1,2,3,4,5,6,7,8,9,10]
    s = slice(5)
    print(s)
    print(a[s])
    
    s2 = slice(2,5)
    print(s2)
    print(a[s2])

# sliceDemo()

def f(a,b,c):
    print(a,b,c)

def jiebaoDemo():
    a = (1,2,3)
    f(*a)
    # f(a)
    
# jiebaoDemo()

def arrDemo():
    a = [1,2,3,4,5,6,7,8,9,10]
    for i in a:
        print(i)

# arrDemo()

def forDemo():
    a = [('afewa',12),('fea',33)]
    print(*[(name,age*2) for name,age in a])
    
forDemo()
    