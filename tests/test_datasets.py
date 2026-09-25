from urllib.error import HTTPError

import pandas as pd
import pytest

from pycaret.datasets import get_data


def test_get_data_remote_uses_single_request(monkeypatch):
    expected = pd.DataFrame({"value": [1, 2]})
    addresses = []

    def read_csv(address):
        addresses.append(address)
        return expected.copy()

    def unexpected_head(*args, **kwargs):
        pytest.fail("Remote datasets should not require a HEAD request")

    monkeypatch.setattr(pd, "read_csv", read_csv)
    monkeypatch.setattr("requests.head", unexpected_head)
    actual = get_data("example", address="https://example.com", verbose=False)

    pd.testing.assert_frame_equal(actual, expected)
    assert addresses == ["https://example.com/example.csv"]


def test_get_data_remote_http_error(monkeypatch):
    def read_csv(address):
        raise HTTPError(address, 404, "Not Found", None, None)

    monkeypatch.setattr(pd, "read_csv", read_csv)
    with pytest.raises(
        ValueError, match="Data could not be read. Please check your inputs"
    ):
        get_data("missing", address="https://example.com", verbose=False)


def test_datasets():
    #########################
    # Load Local File ####
    #########################

    # # loading dataset
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # data = get_data("test_files/dummy_dataset")
    # assert isinstance(data, pd.DataFrame)
    # rows, cols = data.shape
    # assert rows >= 1
    # assert cols >= 1

    ##############################
    # GitHub Common folder ####
    ##############################

    # loading list of datasets
    index = get_data("index")
    assert isinstance(index, pd.DataFrame)
    rows, cols = index.shape
    assert rows > 1
    assert cols == 8

    # loading dataset
    data = get_data("credit")
    assert isinstance(data, pd.DataFrame)
    rows, cols = data.shape
    assert rows == 24000
    assert cols == 24
    assert data.size == 576000

    ################################
    # GitHub Specific folder ####
    ################################

    folder = "time_series/seasonal"
    # loading list of datasets
    index = get_data("index", folder=folder)
    assert isinstance(index, pd.DataFrame)
    rows, cols = index.shape
    assert rows > 1
    assert cols == 12

    # loading dataset
    data = get_data("1", folder=folder)
    assert isinstance(data, pd.DataFrame)
    rows, cols = data.shape
    assert rows >= 1
    assert cols >= 1

    ###########################
    # `sktime` datasets ####
    ###########################

    # loading dataset
    data = get_data("airline")
    assert isinstance(data, pd.Series)
    rows = len(data)
    assert rows >= 1

    ###########################
    # Incorrect dataset ####
    ###########################

    with pytest.raises(ValueError) as errmsg:
        _ = get_data("wrong")

    exceptionmsg = errmsg.value.args[0]
    assert exceptionmsg == "Data could not be read. Please check your inputs..."
