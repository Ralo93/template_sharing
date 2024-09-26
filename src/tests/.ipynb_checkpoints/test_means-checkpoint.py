from src.utils import sum_up
import pytest
import sys
sys.path.append('..')


@pytest.fixture
def sample_data():
    '''
    This creates some dummy data for your tests to run, can be used for doubles, mocks etc.
    '''
    return [1, 2, 3]

def test_calculate_sum(sample_data):
    '''
    Simply use asserts for your unit tests
    '''
    assert sum_up(sample_data) == 6