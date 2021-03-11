from lab3_one_dimensional_unconstrained.func_obj import FuncObj
from lab3_one_dimensional_unconstrained.bisection_method import BisectionMethod

if __name__ == '__main__':
    func_obj_quadratic = FuncObj(lambda x: x**2)
    bisection_optimizer = BisectionMethod([-4, 100], func_obj_quadratic)
    print(bisection_optimizer.get_minimum_point(0.000000000001))
    print(func_obj_quadratic.__dict__)