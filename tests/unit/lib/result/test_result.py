import pytest

from lib.result import Err, Nothing, Ok, Option, Result, Some


def test_ok_reports_ok_and_not_err() -> None:
    result = Ok(1)

    assert result.ok is True
    assert result.err is False


def test_err_reports_err_and_not_ok() -> None:
    result = Err(ValueError('boom'))

    assert result.ok is False
    assert result.err is True


def test_ok_map_transforms_value() -> None:
    assert Ok(1).map(lambda value: value + 1) == Ok(2)


def test_err_map_is_noop() -> None:
    error = ValueError('boom')
    assert Err(error).map(lambda value: value + 1) == Err(error)


def test_ok_and_then_chains() -> None:
    assert Ok(1).and_then(lambda value: Ok(value + 1)) == Ok(2)


def test_err_and_then_short_circuits() -> None:
    error = ValueError('boom')
    assert Err(error).and_then(lambda value: Ok(value + 1)) == Err(error)


def test_ok_unwrap_or_returns_value() -> None:
    assert Ok(1).unwrap_or(0) == 1


def test_err_unwrap_or_returns_default() -> None:
    assert Err(ValueError('boom')).unwrap_or(0) == 0


def test_match_narrows_ok_and_err() -> None:
    def describe(result: Result[int, ValueError]) -> str:
        match result:
            case Ok(value):
                return f'ok:{value}'
            case Err(error):
                return f'err:{error}'

    assert describe(Ok(1)) == 'ok:1'
    assert describe(Err(ValueError('boom'))) == 'err:boom'


def test_some_reports_some_and_not_none() -> None:
    option = Some(1)

    assert option.some is True
    assert option.none is False


def test_nothing_reports_none_and_not_some() -> None:
    option = Nothing()

    assert option.some is False
    assert option.none is True


def test_some_map_transforms_value() -> None:
    assert Some(1).map(lambda value: value + 1) == Some(2)


def test_nothing_map_is_noop() -> None:
    assert Nothing().map(lambda value: value + 1) == Nothing()


def test_some_and_then_chains() -> None:
    assert Some(1).and_then(lambda value: Some(value + 1)) == Some(2)


def test_nothing_and_then_short_circuits() -> None:
    assert Nothing().and_then(lambda value: Some(value + 1)) == Nothing()


def test_some_unwrap_or_returns_value() -> None:
    assert Some(1).unwrap_or(0) == 1


def test_nothing_unwrap_or_returns_default() -> None:
    assert Nothing().unwrap_or(0) == 0


def test_some_ok_or_becomes_ok() -> None:
    assert Some(1).ok_or(ValueError('missing')) == Ok(1)


def test_nothing_ok_or_becomes_err() -> None:
    error = ValueError('missing')
    assert Nothing().ok_or(error) == Err(error)


def test_match_narrows_some_and_nothing() -> None:
    def describe(option: Option[int]) -> str:
        match option:
            case Some(value):
                return f'some:{value}'
            case Nothing():
                return 'nothing'

    assert describe(Some(1)) == 'some:1'
    assert describe(Nothing()) == 'nothing'


def test_unwrap_returns_the_value_of_ok() -> None:
    assert Ok(3).unwrap() == 3


def test_unwrap_raises_the_error_of_err() -> None:
    error = ValueError('nope')

    with pytest.raises(ValueError, match='nope') as raised:
        Err(error).unwrap()

    assert raised.value is error
