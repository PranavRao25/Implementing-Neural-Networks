import pytest
from src.node import *

def test_Addition():
    add = Addition()
    data1 = np.array([1, 2, 3])
    data2 = np.array([3, 4, 5])

    assert np.array_equal(np.array([4, 6, 8]), add(data1, data2))

    grad_out = np.array([1, 2, 3])
    grad1, grad2 = add._backward(grad_out)

    assert data1.shape == data2.shape == grad_out.shape
    assert grad1.shape == grad2.shape == grad_out.shape

def test_Multiplication():
    mul = Multiplication()

    # compatible vectors
    data1 = np.array([1, 2, 3])
    data2 = np.array([3, 4, 5])

    assert np.array_equal(mul(data1, data2).data, np.array([26]))

    # matrix multiplication
    data1 = np.array([[1, 2, 3], [4, 5, 6]])
    data2 = np.array([1, 2])

    # incompatible vectors
    with pytest.raises(Exception) as e_info:
        data2 = np.array([2, 4, 5, 6, 7])
        _ = mul(data1, data2)

    # backward
    data1 = np.array([1, 2, 3])
    data2 = np.array([3, 4, 5])
    grad_out = np.array([5])
    grad1, grad2 = mul._backward(grad_out)

    assert np.array_equal(grad1, data2)
    assert np.array_equal(grad2, data1)

def test_Division():
    pass

def test_Power():
    pass

def test_Tanh():
    pass

def test_ReLU():
    pass

def test_Subtraction():
    pass

def test_Exp():
    pass

def test_MSE():
    pass

def test_MAE():
    pass
